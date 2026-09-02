import os
import json
import uuid
from flask import Flask, request, render_template, send_from_directory, redirect, url_for

from pipeline import process_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:10]
    ext = os.path.splitext(file.filename)[1] or ".png"
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    file.save(upload_path)

    try:
        paths, stats = process_image(upload_path, OUTPUT_DIR, job_id)
        with open(os.path.join(OUTPUT_DIR, f"{job_id}.stats.json"), "w") as f:
            json.dump(stats, f)
    except Exception as e:
        return render_template("index.html", error=str(e))

    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id):
    preview_name = f"{job_id}_preview.png"
    stats_path = os.path.join(OUTPUT_DIR, f"{job_id}.stats.json")
    if not os.path.exists(os.path.join(OUTPUT_DIR, preview_name)) or not os.path.exists(stats_path):
        return redirect(url_for("index"))
    with open(stats_path) as f:
        stats = json.load(f)
    return render_template("result.html", job_id=job_id, preview_name=preview_name, stats=stats)


@app.route("/outputs/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/outputs/<path:filename>/download")
def download_attach(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=True)
