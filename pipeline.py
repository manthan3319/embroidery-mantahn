"""
Core image -> embroidery-file pipeline.

Two strategies, auto-selected per image:
  - "color": for photographed/colorful designs (k-means color quantization)
  - "lineart": for outline/coloring-page style art (flood-fill on enclosed regions)

Both funnel into shared fill/satin stitch generation and a DST/EXP-safe
chained stitch emitter (no single stitch/jump step may exceed the format's
~12.1mm per-step limit, or the design corrupts on import).
"""
import numpy as np
import cv2
import pyembroidery
from pyembroidery import EmbThread

SCALE = 1.2          # 0.1mm units per source-image pixel
SAFE_MAX = 80.0       # max stitch/jump step, in 0.1mm units (hard format limit is 121)
WIDTH_THRESHOLD = 16  # px; region avg width below this -> satin, else tatami fill
MIN_AREA_COLOR = 150
MIN_AREA_LINEART = 60


def detect_pipeline_type(image_bgr):
    """Low saturation over the non-background area => line-art; else colorful photo."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    non_white = gray < 240
    if non_white.sum() < 100:
        return "color"
    median_sat = np.median(hsv[:, :, 1][non_white])
    return "lineart" if median_sat < 30 else "color"


# ---------------------------------------------------------------------------
# Shared stitch generation
# ---------------------------------------------------------------------------

def raster_fill(mask, row_step=4):
    stitches = []
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return stitches
    row_min, row_max = int(rows.min()), int(rows.max())
    direction = 1
    for r in range(row_min, row_max + 1, row_step):
        cols = np.where(mask[r])[0]
        if len(cols) == 0:
            continue
        splits = np.where(np.diff(cols) > 1)[0]
        segments = np.split(cols, splits + 1)
        if direction < 0:
            segments = segments[::-1]
        for seg in segments:
            c0, c1 = int(seg[0]), int(seg[-1])
            if direction > 0:
                stitches.append((c0, r)); stitches.append((c1, r))
            else:
                stitches.append((c1, r)); stitches.append((c0, r))
        direction *= -1
    return stitches


def region_shape_stats(mask):
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) < 6:
        return None
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T) + np.eye(2) * 1e-3
    if np.any(~np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    perp = eigvecs[:, 0]
    proj_p = centered @ principal
    proj_q = centered @ perp
    length = proj_p.max() - proj_p.min()
    if length < 1 or not np.isfinite(length):
        return None
    avg_width = pts.shape[0] / length
    return mean, principal, perp, proj_p, proj_q, length, avg_width


def satin_fill(mask, step=2.4):
    stats_ = region_shape_stats(mask)
    if stats_ is None:
        return []
    mean, principal, perp, proj_p, proj_q, length, avg_width = stats_
    t_min, t_max = proj_p.min(), proj_p.max()
    ts = np.arange(t_min, t_max, step)
    stitches = []
    direction = 1
    for t in ts:
        sel = np.abs(proj_p - t) < step
        if not np.any(sel):
            continue
        qv = proj_q[sel]
        q_min, q_max = qv.min(), qv.max()
        pt1 = mean + principal * t + perp * q_min
        pt2 = mean + principal * t + perp * q_max
        if direction > 0:
            stitches.append((pt1[0], pt1[1])); stitches.append((pt2[0], pt2[1]))
        else:
            stitches.append((pt2[0], pt2[1])); stitches.append((pt1[0], pt1[1]))
        direction *= -1
    return stitches


def generate_stitches(mask):
    stats_ = region_shape_stats(mask)
    if stats_ is not None:
        _, _, _, _, _, length, avg_width = stats_
        if avg_width < WIDTH_THRESHOLD and length > 10:
            s = satin_fill(mask)
            if len(s) >= 4:
                return s, "satin"
    return raster_fill(mask), "fill"


class ChainedEmitter:
    """Ensures no single stitch/jump exceeds SAFE_MAX (DST/EXP format limit)."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.last_pos = None

    def emit(self, cmd, x, y):
        lx, ly = self.last_pos if self.last_pos is not None else (x, y)
        dist = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
        if dist > SAFE_MAX:
            n = int(np.ceil(dist / SAFE_MAX))
            for k in range(1, n + 1):
                ix = lx + (x - lx) * k / n
                iy = ly + (y - ly) * k / n
                self.pattern.add_stitch_absolute(cmd, ix, iy)
        else:
            self.pattern.add_stitch_absolute(cmd, x, y)
        self.last_pos = (x, y)


def build_pattern(regions):
    """regions: list of (color_rgb[3], bool_mask). Returns (pattern, stats_dict)."""
    pattern = pyembroidery.EmbPattern()
    emitter = ChainedEmitter(pattern)
    prev_color_key = None
    kind_counts = {"satin": 0, "fill": 0}
    used = 0
    for color, mask in regions:
        stitches, kind = generate_stitches(mask)
        if len(stitches) < 4:
            continue
        kind_counts[kind] += 1
        color_key = tuple(np.round(color).astype(int))
        if prev_color_key is None:
            thread = EmbThread(); thread.set_color(int(color[0]), int(color[1]), int(color[2]))
            pattern.add_thread(thread)
        elif color_key != prev_color_key:
            pattern.color_change()
            thread = EmbThread(); thread.set_color(int(color[0]), int(color[1]), int(color[2]))
            pattern.add_thread(thread)
        else:
            pattern.trim()
        x0, y0 = stitches[0]
        emitter.emit(pyembroidery.JUMP, x0 * SCALE, y0 * SCALE)
        for (x, y) in stitches[1:]:
            emitter.emit(pyembroidery.STITCH, x * SCALE, y * SCALE)
        prev_color_key = color_key
        used += 1
    pattern.end()
    return pattern, {"regions_used": used, **kind_counts, "total_stitches": len(pattern.stitches)}


# ---------------------------------------------------------------------------
# Strategy 1: colorful photographed design -> k-means color quantization
# ---------------------------------------------------------------------------

def segment_color_design(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    smoothed = cv2.bilateralFilter(image_rgb, d=25, sigmaColor=100, sigmaSpace=15)
    smoothed = cv2.bilateralFilter(smoothed, d=25, sigmaColor=100, sigmaSpace=15)
    smoothed = cv2.medianBlur(smoothed, 5)

    bg_ref = smoothed[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1]].astype(np.float64).mean(axis=0)
    dist_to_bg = np.linalg.norm(smoothed.astype(np.float64) - bg_ref, axis=2)
    is_bg = dist_to_bg < 22.0

    K = 12
    fg_pixels = smoothed[~is_bg].reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, fg_labels, _ = cv2.kmeans(fg_pixels, K, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    fg_labels = fg_labels.flatten()

    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[~is_bg] = fg_labels

    flat_orig = image_rgb.reshape(-1, 3).astype(np.float64)
    fg_orig = flat_orig[(~is_bg).reshape(-1)]
    real_color = {lbl: fg_orig[fg_labels == lbl].mean(axis=0) for lbl in range(K)}

    regions = []
    for lbl in range(K):
        color_mask = (label_map == lbl).astype(np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        for comp_id in range(1, num):
            if stats[comp_id, cv2.CC_STAT_AREA] < MIN_AREA_COLOR:
                continue
            regions.append((real_color[lbl], comp_labels == comp_id))
    return regions


# ---------------------------------------------------------------------------
# Strategy 2: line-art / coloring-page design -> flood-fill enclosed regions
# ---------------------------------------------------------------------------

GREENS = [(76, 153, 68), (102, 170, 90), (60, 120, 50), (90, 140, 60)]
PINKS = [(232, 158, 178), (219, 112, 147), (240, 190, 200), (205, 100, 130), (225, 140, 165)]
GOLD = [(230, 190, 90), (210, 170, 70)]


def _classify_color(mask, area, rng):
    ys, xs = np.where(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1)
    aspect = long_side / short_side
    if area < 130:
        return GOLD[rng.integers(0, len(GOLD))]
    if aspect > 2.2 and area > 250:
        return GREENS[rng.integers(0, len(GREENS))]
    return PINKS[rng.integers(0, len(PINKS))]


def segment_lineart_design(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, line_mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    fillable = cv2.bitwise_not(line_mask)
    num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(fillable, connectivity=4)

    border_labels = set()
    border_labels.update(comp_labels[0, :].tolist())
    border_labels.update(comp_labels[-1, :].tolist())
    border_labels.update(comp_labels[:, 0].tolist())
    border_labels.update(comp_labels[:, -1].tolist())
    border_labels.discard(0)

    rng = np.random.default_rng(7)
    regions = []
    for lbl in range(1, num):
        if lbl in border_labels:
            continue
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < MIN_AREA_LINEART:
            continue
        mask = comp_labels == lbl
        color = np.array(_classify_color(mask, area, rng), dtype=np.float64)
        regions.append((color, mask))
    return regions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_image(image_path, out_dir, base_name):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    kind = detect_pipeline_type(image_bgr)
    if kind == "lineart":
        regions = segment_lineart_design(image_bgr)
    else:
        regions = segment_color_design(image_bgr)

    pattern, stats = build_pattern(regions)
    stats["pipeline"] = kind
    stats["region_count"] = len(regions)

    paths = {}
    for ext, writer in [
        ("exp", pyembroidery.write_exp),
        ("dst", pyembroidery.write_dst),
        ("pes", pyembroidery.write_pes),
    ]:
        p = f"{out_dir}/{base_name}.{ext}"
        writer(pattern, p)
        paths[ext] = p
    for ext in ["col", "inf"]:
        p = f"{out_dir}/{base_name}.{ext}"
        pyembroidery.write(pattern, p)
        paths[ext] = p

    preview_path = f"{out_dir}/{base_name}_preview.png"
    pyembroidery.write_png(pattern, preview_path)
    paths["preview"] = preview_path

    return paths, stats
