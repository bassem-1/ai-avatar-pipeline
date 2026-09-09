# AI Avatar Video Pipeline — Setup & Reference Guide

A complete guide to generating personalized talking-head videos using your own voice and face, entirely with open-source tools and free Google Colab GPU.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Recording Your Reference Video](#recording-your-reference-video)
3. [Step-by-Step Colab Setup](#step-by-step-colab-setup)
4. [Model Details](#model-details)
5. [Cost & Time Estimates](#cost--time-estimates)
6. [Troubleshooting](#troubleshooting)
7. [Next Improvements & Roadmap](#next-improvements--roadmap)

---

## Pipeline Overview

```
Your reference video (1 min of you speaking)
        |
        |-- ffmpeg --> reference_voice.wav   (30s audio clip)
        |-- ffmpeg --> reference_face.jpg    (one frame)
        |
        v
   XTTS v2 (Coqui)
   Voice cloning from reference_voice.wav
   Text script --> generated_speech.wav
        |
        v
   SadTalker
   reference_face.jpg + generated_speech.wav --> avatar_video.mp4
        |
        v
   GFPGAN (face enhancement)
        |
        v
   Final output: talking-head MP4 of you saying your script
```

### Technologies Used

| Component | Tool | License |
|---|---|---|
| Voice cloning & TTS | Coqui XTTS v2 | Mozilla Public License 2.0 |
| Talking-head / lip sync | SadTalker | MIT |
| Face enhancement | GFPGAN | Apache 2.0 |
| Audio/video processing | FFmpeg | LGPL 2.1+ |
| Runtime | Google Colab (T4 GPU) | Free tier available |

No API keys. No paid subscriptions. No proprietary models.

---

## Recording Your Reference Video

The quality of your reference video is the single biggest factor in the quality of the output. Spend 5 minutes getting this right.

### Quick Checklist

- [ ] Quiet room — no background music, TV, HVAC noise, or echo
- [ ] Good lighting — face evenly lit, no harsh shadows across your face
- [ ] Camera at eye level — not looking up or down at the lens
- [ ] Face centered and clearly visible throughout
- [ ] No sunglasses, masks, or large occlusions
- [ ] Speak naturally — normal conversational pace and tone
- [ ] Duration: 30 seconds minimum, 1–2 minutes is ideal
- [ ] Format: MP4, MOV, AVI, or MKV
- [ ] Resolution: at least 720p (1080p preferred)

### Tips for Best Voice Cloning

XTTS v2 learns your voice characteristics from the reference audio. To get the best clone:

- Speak a variety of sentences — mix short and long ones
- Include natural pauses and intonation changes
- Avoid whispering, shouting, or exaggerated accents
- Read aloud for ~1 minute if you are not sure what to say (read a news article, a book passage, or describe your day)
- The audio will be extracted as a 30-second clip. Make sure seconds 5–35 are clean and representative of your natural speaking voice

### Tips for Best Face Animation

SadTalker generates lip sync from a single still frame extracted from your video.

- The frame is extracted at the 5-second mark by default
- Make sure your face is fully visible and well-lit at that point
- Front-facing is best. Up to ~30 degrees of side angle is acceptable
- Neutral expression or slight smile works well
- Avoid blinking or talking during the exact frame that gets extracted (if you see an issue in the preview, re-run the extraction cell with a different timestamp)

### Example Script to Record

If you are not sure what to say during your reference video, read this aloud at a natural pace:

> "The quick brown fox jumps over the lazy dog. I am recording this video as a reference for my AI avatar system. The system will use my voice to generate new speech. I want to make sure it captures my natural speaking style. I speak at a comfortable pace, neither too fast nor too slow. I hope this recording gives the model enough information to replicate my voice accurately."

---

## Step-by-Step Colab Setup

### 1. Open the Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `AI_Avatar_Pipeline.ipynb`

### 2. Enable GPU Runtime

**This step is mandatory.** The pipeline will not run without a GPU.

1. Click **Runtime** in the top menu
2. Click **Change runtime type**
3. Set **Hardware accelerator** to **GPU**
4. Set **GPU type** to **T4** (free) or **A100** (Colab Pro, faster)
5. Click **Save**

You should see a green GPU indicator in the top right of the notebook.

### 3. Run the Notebook

Run each section in order by clicking the play button on each cell, or use **Runtime → Run all**.

**Do not skip sections.** Each section depends on the previous one.

| Section | What it does | Approx. time |
|---|---|---|
| Section 1: Environment Setup | Installs system packages, clones SadTalker, installs TTS | 5–8 min (first run) |
| Section 2: Download Checkpoints | Downloads SadTalker and GFPGAN model weights | 3–5 min (first run) |
| Section 3: Upload Reference Material | Upload video, extract audio + face | 1–2 min |
| Section 4: Generate Video | Voice synthesis + face animation | 5–15 min |

### 4. Upload Your Reference Video

When Cell 3.1 runs, a file picker will appear. Select your recorded reference video. Upload size limit in Colab is typically 100 MB for the free tier — if your video is larger, compress it first with:

```bash
ffmpeg -i input.mp4 -crs 28 -preset fast output_compressed.mp4
```

### 5. Edit Your Script

In Cell 4.1, replace the example `SCRIPT` variable with the text you want your avatar to say. Save and run the cell.

### 6. Download Your Video

Cell 4.4 will automatically download the generated MP4 to your computer.

---

## Model Details

### XTTS v2 (Coqui TTS)

- **Full name**: Coqui TTS — XTTS v2 (Cross-lingual Text-to-Speech version 2)
- **Model size**: ~1.8 GB
- **What it does**: Given a short audio clip of a speaker, it clones that voice and can synthesize new speech in that voice from any text input
- **Languages**: 17 languages — English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Korean, Hungarian, Hindi
- **Minimum reference audio**: 6 seconds (30 seconds recommended for better quality)
- **Output sample rate**: 24 kHz
- **License**: Mozilla Public License 2.0
- **GitHub**: https://github.com/coqui-ai/TTS
- **Key parameter**: `language="en"` — change to match the language your script is in

### SadTalker

- **Full name**: SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation
- **What it does**: Takes a static face image and a driving audio file, and outputs a video of the face lip-synced to the audio with realistic head motion
- **Checkpoint sizes**: ~600 MB total
- **Inference modes**:
  - `--size 256`: 256x256 output, faster, less VRAM
  - `--size 512`: 512x512 output, slower, more VRAM (requires ~8 GB VRAM)
  - `--still`: Reduces head motion (more professional/static look)
  - `--preprocess full`: Processes full face area (better quality)
- **License**: MIT
- **GitHub**: https://github.com/OpenTalker/SadTalker
- **Paper**: https://arxiv.org/abs/2211.12194

### GFPGAN (Face Enhancer)

- **Full name**: GAN-based Face Restoration with Perception-oriented Losses (GAN Prior)
- **What it does**: Restores facial details in low-quality or synthesized face images/videos. Applied by SadTalker as a post-processing step
- **Version used**: GFPGANv1.4
- **License**: Apache 2.0
- **GitHub**: https://github.com/TencentARC/GFPGAN
- **Why it matters**: SadTalker's raw output can look slightly blurry or artificial around the mouth area. GFPGAN significantly sharpens and restores facial texture

---

## Cost & Time Estimates

### Google Colab Free Tier (T4 GPU)

| Task | Time estimate |
|---|---|
| First-time setup (install + download) | 8–13 minutes |
| Subsequent runs (setup already done) | 1–2 minutes |
| XTTS voice synthesis (per minute of speech) | ~1–2 minutes |
| SadTalker inference (per minute of video) | ~3–5 minutes |

**Practical examples:**

| Script length | Total generation time |
|---|---|
| 30 seconds | ~4–7 minutes |
| 1 minute | ~5–10 minutes |
| 2 minutes | ~8–15 minutes |
| 5 minutes | ~20–35 minutes |

### Google Colab Pro / Pro+ (A100 GPU)

Roughly 3–4x faster than T4. A 2-minute video takes ~4–6 minutes total. Recommended for frequent use or longer scripts.

### Cost

- **Free tier**: $0. Limited to ~12 hours of GPU time per day, sessions may disconnect after 90 minutes of inactivity
- **Colab Pro**: ~$10/month. Longer sessions, priority access to better GPUs
- **Colab Pro+**: ~$50/month. Background execution, A100 access, more compute units
- **All models are free**: No usage fees for XTTS, SadTalker, or GFPGAN

---

## Troubleshooting

### Voice sounds robotic or nothing like me

**Causes and fixes:**

1. **Reference audio too short** — Use at least 30 seconds of clean speech. The pipeline extracts 30s starting at 5s; make sure that window is good audio.
2. **Background noise in reference** — Record in a quieter environment. Even mild HVAC noise degrades cloning quality.
3. **Wrong language setting** — If your script is in a language other than English, change `language="en"` to the correct code (e.g., `"de"` for German, `"ar"` for Arabic).
4. **Try a different segment** — In Cell 3.2, change `-ss 5` to `-ss 10` or `-ss 15` to extract audio from a different part of the video.
5. **Normalize the reference audio** — Add `-af loudnorm` to the ffmpeg command for more consistent volume.

### Face not detected / SadTalker fails immediately

**Causes and fixes:**

1. **Poor reference frame** — The frame at 5s may have eyes closed, be blurry, or partially occluded. Re-run Cell 3.2 with `-ss 3` or `-ss 8` to try a different moment.
2. **Profile or extreme angle** — Use a more front-facing reference.
3. **Small face in frame** — If your face only takes up a small part of the frame, crop the image: add `-vf "crop=iw*0.5:ih*0.5:iw*0.25:ih*0.25"` to the ffmpeg face extraction command (adjust values to center on your face).
4. **Glasses** — Strong reflections from glasses sometimes confuse the face detector. Try a frame without glare.

### Out of Memory (CUDA OOM)

**Fixes in order of simplicity:**

1. Confirm you are using `--size 256` (already the default in the notebook)
2. Go to **Runtime → Restart runtime**, then rerun only Section 4 (checkpoints and installs persist within the same session but are lost on restart — you may need to re-download)
3. Shorten your script to reduce audio length
4. Upgrade to Colab Pro for A100 access (24 GB VRAM)
5. Use `--preprocess crop` instead of `--preprocess full` to reduce memory usage

### SadTalker takes forever or hangs

1. Run `!nvidia-smi` in a new cell to check GPU utilization
2. If GPU shows 0% utilization for more than 2 minutes, the process may be stuck — restart the runtime
3. Check that `OUTPUT_AUDIO` path is correct and the file exists: `!ls -lh /content/workspace/generated_speech.wav`

### TTS import error or version conflict

```python
# Run this in a new cell to reinstall cleanly
!pip install --upgrade TTS
import importlib, TTS
importlib.reload(TTS)
```

### Video downloads but won't play

The video is encoded as H.264 MP4, which plays in all major media players. If it doesn't play:

1. Try VLC media player (free, plays everything)
2. The file may be corrupted if the download was interrupted — re-run Cell 4.4

### Colab session disconnects during generation

Colab disconnects after 90 minutes of inactivity on the free tier. To avoid losing progress:

1. Keep the browser tab active (avoid switching away for long periods)
2. Mount Google Drive and save intermediate outputs after each section
3. Use Colab Pro for longer session limits

---

## Next Improvements & Roadmap

The current pipeline is functional and produces good results. Here are the most impactful upgrades to consider, roughly in order of impact vs. effort:

### Higher Quality Lip Sync — LatentSync (Meta, 2024)

LatentSync is a newer lip-sync model from Meta Research that operates in the latent space of a video diffusion model. It produces significantly more natural lip movements and better temporal consistency than SadTalker, especially for complex phonemes.

- **Paper**: LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync
- **Why**: More realistic mouth movements, fewer artifacts, better generalization to challenging head poses
- **Trade-off**: Slower inference, requires more VRAM, less mature tooling than SadTalker
- **GitHub**: https://github.com/bytedance/LatentSync

### Better Voice Quality — F5-TTS or CosyVoice

XTTS v2 is solid but newer models produce more natural-sounding output:

- **F5-TTS**: Flow-matching based TTS with excellent naturalness. Can zero-shot clone a voice from a short clip. Particularly good for English.
  - GitHub: https://github.com/SWivid/F5-TTS
- **CosyVoice 2** (Alibaba): High-quality multilingual TTS with strong voice cloning. Particularly strong for Chinese and other Asian languages.
  - GitHub: https://github.com/FunAudioLLM/CosyVoice
- **Why consider switching**: Lower word error rate, more natural prosody (rhythm and intonation), better handling of numbers and abbreviations

### Fine-Tune XTTS on More Data

For a significantly better voice match, fine-tune the XTTS v2 model on 5–30 minutes of your own speech data. Coqui provides a fine-tuning script.

- Collect 5–30 minutes of clean audio of yourself speaking
- Run the XTTS fine-tuning script on Colab or locally
- The fine-tuned model will capture your voice characteristics far more accurately
- Guide: https://docs.coqui.ai/en/latest/fine_tuning.html

### Composite Onto Original Background

Currently the output is a cropped/resized talking head on a plain background. A more polished result composites the talking head back onto the original video background:

1. Extract background from reference video (or a clean background video)
2. Segment the talking head using a matting model (e.g., MODNet or RVM)
3. Composite using ffmpeg or OpenCV
4. Result: avatar appears to be sitting in your original environment

### Batch Generation for Multiple Scripts

Generate multiple videos in sequence without re-uploading reference material:

- Define a list of scripts as a Python list
- Loop through each, calling XTTS and SadTalker
- Save each output with a unique filename
- Useful for creating a series of short video segments or social media content

### Resolution Upscaling with Real-ESRGAN

SadTalker at 256x256 is low resolution for modern screens. Real-ESRGAN can upscale the output to 1024x1024 or higher while preserving sharpness:

- Run Real-ESRGAN on the output video frames
- 4x upscale (256 → 1024) takes ~1–2 minutes on T4
- GitHub: https://github.com/xinntao/Real-ESRGAN
- Command: `python inference_realesrgan_video.py -n RealESRGAN_x4plus -i input.mp4 -o output_4x.mp4`

### Audio Post-Processing

Improve the perceived naturalness of the generated speech:

- **Noise shaping**: Add very slight room reverb to make it sound less dry
- **EQ**: Apply a gentle high-pass filter to remove low-frequency artifacts
- **Compression**: Normalize dynamics so the voice sounds more broadcast-quality
- Tools: `pedalboard` (Python library by Spotify), `sox`, or `ffmpeg -af` filters

---

## File Structure After Running

```
/content/
├── SadTalker/
│   ├── checkpoints/           # Model weights (~600 MB)
│   │   ├── mapping_00109-model.pth.tar
│   │   ├── mapping_00229-model.pth.tar
│   │   ├── SadTalker_V0.0.2_256.safetensors
│   │   └── SadTalker_V0.0.2_512.safetensors
│   └── gfpgan/weights/        # GFPGAN weights (~340 MB)
│       ├── GFPGANv1.4.pth
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
└── workspace/
    ├── your_video.mp4         # Your uploaded reference video
    ├── reference_voice.wav    # Extracted 30s voice reference
    ├── reference_face.jpg     # Extracted face frame
    ├── generated_speech.wav   # XTTS output audio
    └── sadtalker_output/
        └── *.mp4              # Final avatar video
```

---

## Resources & Credits

- **Coqui TTS (XTTS v2)**: https://github.com/coqui-ai/TTS
- **SadTalker**: https://github.com/OpenTalker/SadTalker — Wenxuan Zhang et al., CVPR 2023
- **GFPGAN**: https://github.com/TencentARC/GFPGAN — Xintao Wang et al.
- **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN
- **LatentSync**: https://github.com/bytedance/LatentSync
- **F5-TTS**: https://github.com/SWivid/F5-TTS
- **CosyVoice**: https://github.com/FunAudioLLM/CosyVoice
