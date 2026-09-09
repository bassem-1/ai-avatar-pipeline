"""
AI Avatar — Flask Backend
Wraps XTTS v2 (voice cloning) + SadTalker (talking-head) pipeline.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.

Environment variables:
    SADTALKER_DIR  Path to the cloned SadTalker repo (default: ./SadTalker)
    PORT           Port to listen on (default: 5000)
"""
import glob
import os
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SADTALKER_DIR = Path(os.getenv("SADTALKER_DIR", "SadTalker"))
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi",
]

# ---------------------------------------------------------------------------
# Job state  (in-memory; fine for single-user local use)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ---------------------------------------------------------------------------
# TTS model — loaded once on first use
# ---------------------------------------------------------------------------

_tts_model = None
_tts_lock = threading.Lock()


def _load_tts():
    global _tts_model
    if _tts_model is not None:
        return _tts_model
    with _tts_lock:
        if _tts_model is None:
            import torch
            from TTS.api import TTS

            os.environ["COQUI_TOS_AGREED"] = "1"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts_model


# ---------------------------------------------------------------------------
# Pipeline worker (runs in a background thread)
# ---------------------------------------------------------------------------

def _run_pipeline(
    job_id: str,
    audio_path: Path,
    image_path: Path,
    script: str,
    language: str,
) -> None:
    try:
        # --- Step 1: Clone voice & synthesise speech ------------------
        _update_job(job_id, status="synthesising_voice", progress=20,
                    message="Cloning your voice and synthesising speech…")

        tts = _load_tts()
        generated_audio = UPLOAD_DIR / f"{job_id}_speech.wav"
        tts.tts_to_file(
            text=script,
            speaker_wav=str(audio_path),
            language=language,
            file_path=str(generated_audio),
        )

        # --- Step 2: Animate talking head ----------------------------
        _update_job(job_id, status="generating_video", progress=55,
                    message="Animating your face with SadTalker…")

        if not SADTALKER_DIR.exists():
            raise FileNotFoundError(
                f"SadTalker not found at {SADTALKER_DIR}. "
                "Clone it: git clone https://github.com/OpenTalker/SadTalker.git"
            )

        result_dir = OUTPUT_DIR / job_id
        result_dir.mkdir(exist_ok=True)

        proc = subprocess.run(
            [
                "python", "inference.py",
                "--driven_audio", str(generated_audio),
                "--source_image", str(image_path),
                "--result_dir", str(result_dir),
                "--still",
                "--preprocess", "full",
                "--enhancer", "gfpgan",
                "--size", "256",
            ],
            cwd=str(SADTALKER_DIR),
            capture_output=True,
            text=True,
            timeout=900,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"SadTalker failed (exit {proc.returncode}):\n{proc.stderr[-600:]}"
            )

        # --- Locate output video ------------------------------------
        videos = sorted(
            glob.glob(str(result_dir / "**" / "*.mp4"), recursive=True),
            key=os.path.getmtime,
        )
        if not videos:
            raise FileNotFoundError(
                "SadTalker completed but produced no MP4. "
                "Check that the face image contains a detectable face."
            )

        _update_job(job_id, status="done", progress=100,
                    message="Your video is ready!", output_path=videos[-1])

    except Exception as exc:
        _update_job(job_id, status="error", progress=0, message=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    audio_file = request.files.get("audio")
    image_file = request.files.get("image")
    script = request.form.get("script", "").strip()
    language = request.form.get("language", "en")

    if not audio_file:
        return jsonify({"error": "No voice recording provided."}), 400
    if not image_file:
        return jsonify({"error": "No face image provided."}), 400
    if not script:
        return jsonify({"error": "Script is empty."}), 400
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'."}), 400

    job_id = uuid.uuid4().hex[:10]
    audio_path = UPLOAD_DIR / f"{job_id}_voice.wav"
    image_path = UPLOAD_DIR / f"{job_id}_face.jpg"

    audio_file.save(audio_path)
    image_file.save(image_path)

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": 5, "message": "Job queued…"}

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, audio_path, image_path, script, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    # Don't leak the server-side file path to the client
    return jsonify({k: v for k, v in job.items() if k != "output_path"})


@app.route("/result/<job_id>")
def job_result(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    if job.get("status") != "done":
        return jsonify({"error": "Result not ready yet."}), 425

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "Output file is missing on the server."}), 500

    return send_file(output_path, mimetype="video/mp4", as_attachment=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting AI Avatar server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
