# AI Avatar Pipeline

Generate a talking-head video of yourself speaking any script — using your own voice and face. Fully open-source, no API keys required.

```
Text script
    │
    ▼
XTTS v2 ──── your voice reference ──▶ generated_speech.wav
    │
SadTalker ── your face photo ────────▶ avatar.mp4
    │
GFPGAN (face enhancement)
```

---

## Quick Start — Google Colab (recommended)

1. Open [`AI_Avatar_Pipeline.ipynb`](AI_Avatar_Pipeline.ipynb) in Google Colab
2. **Runtime → Change runtime type → GPU** (T4 free tier works)
3. Run all cells top to bottom
4. Upload your reference video when prompted, enter your script, download your video

---

## Quick Start — Local Web App

### Prerequisites

- Python 3.10+
- GPU strongly recommended (CPU will work but is very slow)
- [SadTalker](https://github.com/OpenTalker/SadTalker) cloned alongside this repo

```bash
# 1. Clone SadTalker into this directory
git clone https://github.com/OpenTalker/SadTalker.git

# 2. Install web app dependencies
pip install -r requirements_web.txt

# 3. Install SadTalker dependencies
pip install -r SadTalker/requirements.txt

# 4. Download model checkpoints (run once)
python - <<'EOF'
import os
from huggingface_hub import hf_hub_download

os.makedirs("SadTalker/checkpoints", exist_ok=True)
for f in ["mapping_00109-model.pth.tar", "mapping_00229-model.pth.tar",
          "SadTalker_V0.0.2_256.safetensors", "SadTalker_V0.0.2_512.safetensors"]:
    hf_hub_download("vinthony/SadTalker", f, local_dir="SadTalker/checkpoints")

os.makedirs("SadTalker/gfpgan/weights", exist_ok=True)
import urllib.request
urls = {
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}
for name, url in urls.items():
    urllib.request.urlretrieve(url, f"SadTalker/gfpgan/weights/{name}")
EOF

# 5. Start the server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Connecting the Web App to Google Colab

If you want to use the Colab GPU but the local web UI, add these cells at the end of the Colab notebook:

```python
# Install server dependencies
!pip install -q flask pyngrok

# Start Flask app from GitHub
!wget -q https://raw.githubusercontent.com/bassem-1/test/main/app.py -O /content/app.py

import threading, subprocess
flask_thread = threading.Thread(
    target=lambda: subprocess.run(["python", "/content/app.py"]), daemon=True
)
flask_thread.start()

import time; time.sleep(3)

from pyngrok import ngrok
url = ngrok.connect(5000)
print(f"\n>>> Paste this URL into the web app's Backend URL field:\n{url}\n")
```

Then paste the printed ngrok URL into the **Backend URL** field at the top of the web app.

---

## Web Interface

| Feature | Details |
|---|---|
| Voice recording | In-browser mic recording with live waveform |
| Voice upload | Upload any audio or video file |
| Camera capture | Take a live selfie via webcam |
| Photo upload | Upload any face photo |
| Language | 17 languages via XTTS v2 |
| Live progress | Real-time status updates during generation |
| Preview | In-browser video playback when done |
| Download | One-click MP4 download |

---

## Tech Stack

| Component | Model | Size | License |
|---|---|---|---|
| Voice cloning | [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2) | ~1.8 GB | CPML |
| Talking head | [SadTalker](https://github.com/OpenTalker/SadTalker) | ~600 MB | MIT |
| Face enhancement | [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) | ~340 MB | Apache-2.0 |
| Web backend | Flask 3 | — | BSD |
| Inference (Colab) | NVIDIA T4 (free) / A100 (Pro) | — | — |

---

## File Structure

```
.
├── AI_Avatar_Pipeline.ipynb   # Colab notebook (main pipeline)
├── app.py                     # Flask web backend
├── templates/
│   └── index.html             # Web UI (voice, camera, preview)
├── requirements_web.txt       # Web app Python deps
├── SETUP_GUIDE.md             # Detailed setup & troubleshooting
└── .gitignore
```

---

## Tips for Best Results

- **Voice**: Record 20–60 s of clear speech with no background noise
- **Face**: Front-facing, even lighting, no glasses or hat
- **Script**: 50–200 words for fastest turnaround; natural punctuation improves prosody
- **Quality**: Change `--size 256` → `--size 512` in the notebook for higher resolution (needs more VRAM)

---

## Roadmap

- [ ] LatentSync (Meta) for higher-quality lip sync
- [ ] F5-TTS for improved voice naturalness
- [ ] Background compositing (talk in your original video background)
- [ ] Batch generation from a CSV of scripts
- [ ] Real-ESRGAN upscaling to 1080p
