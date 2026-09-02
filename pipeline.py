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
from skimage.morphology import skeletonize

SCALE = 1.2          # 0.1mm units per source-image pixel
SAFE_MAX = 80.0       # max stitch/jump step, in 0.1mm units (hard format limit is 121)
WIDTH_THRESHOLD = 16  # px; region avg width below this -> satin, else tatami fill
MIN_AREA_COLOR = 150
MIN_AREA_LINEART = 60


def detect_pipeline_type(image_bgr):
    """Route to "color" (k-means, multi-color photo/graphic), "lineart"
    (thin outline strokes enclosing fillable areas -- a coloring-page style
    drawing), or "silhouette" (a solid-filled dark shape/logo -- the dark
    area itself IS the design to stitch, not an outline around something
    else). Getting line-art vs silhouette wrong is not a small miss: the
    line-art path only fills white pockets ENCLOSED by ink, so if the ink
    itself is the thick solid subject (a logo, a monogram) almost nothing
    survives -- see the gada+text logo case that returned a single 21-stitch
    region.

    Must isolate real foreground before measuring saturation: a plain
    gray<240 cutoff over the whole frame lets a textured/woven fabric
    background (e.g. beige linen, median gray ~222) dominate the pixel
    count and drag the median saturation down, misrouting a colorful
    photographed design into the line-art path. Reuse the same
    corner-sampled background-distance test segment_color_design relies
    on so the saturation check only looks at genuine foreground pixels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = image_bgr.shape[:2]
    sm = cv2.medianBlur(image_bgr, 5)
    bg_ref = sm[[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1]].astype(np.float64).mean(axis=0)
    fg = np.linalg.norm(sm.astype(np.float64) - bg_ref, axis=2) >= 22.0
    if fg.sum() < 100:
        return "color"
    median_sat = np.median(hsv[:, :, 1][fg])
    if median_sat >= 30:
        return "color"

    # Grayscale/low-saturation design: distinguish thin outline strokes from a
    # solid silhouette by how THICK the dark ink is. A coloring-page outline
    # stays a few px wide throughout; a logo/silhouette has large interior
    # areas far from any edge. distanceTransform gives, per ink pixel, the
    # distance to the nearest non-ink pixel -- half the local stroke width.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
    if ink.sum() < 100:
        return "lineart"
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    thick_fraction = (dist > 6).sum() / (ink > 0).sum()
    return "silhouette" if thick_fraction > 0.12 else "lineart"


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


def satin_fill_pca(mask, step=2.4):
    """Single global PCA axis, sliced perpendicular at regular intervals.
    Clean and reliable for a simple blob/ribbon region (a petal, a crown
    decoration) -- the common case from the color and line-art pipelines.
    Not used for branching shapes; see satin_fill for those.
    """
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


def _skeleton_branches(mask):
    """Decompose a mask's morphological skeleton into ordered branches: paths
    between endpoints (degree 1) and junctions (degree >=3). A branching
    skeleton (e.g. connected calligraphy letters, which join at a shirorekha)
    can't be walked as one path, so each branch becomes its own satin pass.
    """
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) < 2:
        return []
    coords = set(zip(xs.tolist(), ys.tolist()))

    def neighbors(p):
        x, y = p
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in coords:
                    out.append(q)
        return out

    degree = {p: len(neighbors(p)) for p in coords}
    branch_points = {p for p, d in degree.items() if d != 2}
    if not branch_points:
        branch_points = {next(iter(coords))}  # closed loop, no natural start

    visited_edges = set()
    branches = []
    for start in branch_points:
        for nb in neighbors(start):
            if (start, nb) in visited_edges:
                continue
            path = [start, nb]
            visited_edges.add((start, nb)); visited_edges.add((nb, start))
            prev, cur = start, nb
            while cur not in branch_points:
                nxts = [n for n in neighbors(cur) if n != prev and (cur, n) not in visited_edges]
                if not nxts:
                    break
                nxt = nxts[0]
                visited_edges.add((cur, nxt)); visited_edges.add((nxt, cur))
                path.append(nxt)
                prev, cur = cur, nxt
            arc_len = sum(
                ((path[i][0] - path[i-1][0]) ** 2 + (path[i][1] - path[i-1][1]) ** 2) ** 0.5
                for i in range(1, len(path))
            )
            # Skeletonizing a real (slightly wavy) boundary throws off short
            # spurious spurs near the main branch -- a well-known medial-axis
            # artifact. Without pruning these, each spur becomes its own tiny
            # noisy zigzag on top of the real shape. Require real length.
            if arc_len >= 25:
                branches.append(path)
    return branches


def satin_fill_skeleton(mask, step_px=3):
    """Skeleton/centerline-based satin: at each point along the shape's
    medial axis, cast a ray perpendicular to the LOCAL tangent out to the
    mask edge on each side (the two rail points), then zigzag between rails.
    Unlike a single global PCA axis, this follows curves correctly and
    handles branching shapes (returns one stitch-group per branch, each
    satin-stitched independently) instead of producing one misleading
    "average" direction for the whole shape.
    """
    h, w = mask.shape
    groups = []
    for branch in _skeleton_branches(mask):
        pts = np.array(branch, dtype=np.float64)
        n = len(pts)
        rails = []
        for idx in range(0, n, max(1, step_px)):
            i0, i1 = max(0, idx - 2), min(n - 1, idx + 2)
            tangent = pts[i1] - pts[i0]
            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                continue
            tangent = tangent / norm
            perp = np.array([-tangent[1], tangent[0]])
            p = pts[idx]

            def march(direction, origin=p):
                pos = origin.copy()
                last_inside = pos.copy()
                for _ in range(60):  # up to ~30px out
                    pos = pos + direction * 0.5
                    xi, yi = int(round(pos[0])), int(round(pos[1]))
                    if 0 <= xi < w and 0 <= yi < h and mask[yi, xi]:
                        last_inside = pos.copy()
                    else:
                        break
                return last_inside

            rails.append((march(perp), march(-perp)))

        if len(rails) < 2:
            continue
        stitches = []
        direction = 1
        for r1, r2 in rails:
            if direction > 0:
                stitches.append((r1[0], r1[1])); stitches.append((r2[0], r2[1]))
            else:
                stitches.append((r2[0], r2[1])); stitches.append((r1[0], r1[1]))
            direction *= -1
        if len(stitches) >= 4:
            groups.append(stitches)
    return groups


def local_thickness(mask):
    """80th-percentile local full-thickness via distance transform -- robust
    to complex/branching shapes (e.g. connected calligraphy strokes) where a
    simple area/(PCA-axis length) ratio is misleading: a long connected word
    has a small average width by that formula only when it happens to be a
    clean ribbon, but a multi-letter blob's bounding-axis length has little
    to do with how thick any individual stroke actually is.

    Uses a high percentile, not the median: EVERY shape (thin or wide) has
    pixels close to its boundary, so the median is dragged down by edge-
    adjacent pixels regardless of whether the shape has real bulk. Only a
    genuinely wide region has a deep, far-from-any-edge core reaching a high
    distance value -- that's what should decide fill vs satin.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    vals = dist[mask]
    return 2 * np.percentile(vals, 80) if len(vals) else 0.0


def generate_stitches(mask, skeleton_satin=False):
    """Returns (stitch_groups, kind): a list of point-lists, one JUMP+STITCH
    pass per group. Fill and PCA-satin are always one group; skeleton-satin
    is one group per skeleton branch (see satin_fill_skeleton).

    skeleton_satin picks WHICH satin algorithm gets a chance, not just a
    threshold: skeleton-based satin follows local curvature and handles
    branching (needed for connected calligraphy strokes in a logo), but on
    a rounder/blobbier region (a petal, a flower centre) the skeleton is a
    multi-armed star rather than a ribbon and renders as scribble noise.
    PCA satin stays the default for the color/line-art pipelines, where
    regions are typically simple blobs; the silhouette pipeline (logos,
    text) opts into skeleton satin instead.

    Satin also needs BOTH "thin" and "elongated": a small round blob can
    have low local_thickness just by being small. Gate on length being well
    beyond the thickness too, so only genuinely ribbon/stroke-shaped
    regions attempt satin at all; small round bits fall through to fill.
    """
    stats_ = region_shape_stats(mask)
    if stats_ is not None:
        _, _, _, _, _, length, avg_width = stats_
        thickness = local_thickness(mask)
        if thickness < WIDTH_THRESHOLD and length > max(10, 2.5 * thickness):
            if skeleton_satin:
                groups = satin_fill_skeleton(mask)
            else:
                s = satin_fill_pca(mask)
                groups = [s] if len(s) >= 4 else []
            if groups:
                return groups, "satin"
    return [raster_fill(mask)], "fill"


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


def build_pattern(regions, skeleton_satin=False):
    """regions: list of (color_rgb[3], bool_mask). Returns (pattern, stats_dict)."""
    pattern = pyembroidery.EmbPattern()
    emitter = ChainedEmitter(pattern)
    prev_color_key = None
    kind_counts = {"satin": 0, "fill": 0}
    used = 0
    for color, mask in regions:
        groups, kind = generate_stitches(mask, skeleton_satin=skeleton_satin)
        groups = [g for g in groups if len(g) >= 4]
        if not groups:
            continue
        kind_counts[kind] += 1
        color_key = tuple(np.round(color).astype(int))
        for stitches in groups:
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

    # Cluster in LAB with lightness heavily downweighted. A photographed satin-stitch
    # design has real light/dark shading WITHIN a single petal/element (thread sheen,
    # stitch-direction reflectance) -- clustering on raw RGB/lightness splits one
    # element into concentric shade bands, which the satin-fill step then renders as
    # nested rings instead of a clean fill. Weighting hue/chroma (a*, b*) over
    # lightness (L*) keeps same-color shading together as one region.
    K = 12
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 0.25
    fg_pixels = lab[~is_bg].reshape(-1, 3)
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
# Strategy 3: solid silhouette / logo -> the dark shape itself is the design
# ---------------------------------------------------------------------------

def segment_silhouette_design(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, ink = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
    # No closing here (unlike the other strategies): a silhouette's thin white
    # gaps -- the bands between concentric mace rings, the counters inside
    # letters -- are real negative space in the source art, not noise. Closing
    # them would fuse rings/letters into a featureless blob.

    num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)

    flat_rgb = image_rgb.reshape(-1, 3).astype(np.float64)
    regions = []
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] < MIN_AREA_LINEART:
            continue
        mask = comp_labels == lbl
        color = flat_rgb[mask.reshape(-1)].mean(axis=0)  # real sampled color, not hardcoded black
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
    elif kind == "silhouette":
        regions = segment_silhouette_design(image_bgr)
    else:
        regions = segment_color_design(image_bgr)

    pattern, stats = build_pattern(regions, skeleton_satin=(kind == "silhouette"))
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
