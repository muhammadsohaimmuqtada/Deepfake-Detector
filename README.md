<div align="center">

# 🔍 Deepfake Detector

**A production-ready, enterprise-grade deepfake detection pipeline.**  
Built on mathematical frequency analysis (FFT) — no black boxes, no GPU required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)](https://mediapipe.dev)
[![React](https://img.shields.io/badge/React-19%2B-61dafb?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview — The Problem & Solution](#overview)
- [Architecture](#architecture)
- [Why This Approach Is Trusted](#why-trusted)
- [Repository Structure](#repository-structure)
- [Backend Setup (Python Detection Engine)](#backend-setup)
- [Frontend Setup (Web UI)](#frontend-setup)
- [Usage](#usage)
- [Output — Forensic Report](#output)
- [Roadmap](#roadmap)

---

## Overview

### The Problem

Synthetic media — AI-generated "deepfake" videos — represent one of the most significant threats to information integrity today. Legacy detection systems rely on monolithic neural networks that:

- **Act as black boxes**, offering no explainable evidence for why a video was flagged.
- **Require expensive GPU infrastructure**, making them inaccessible to most organizations.
- **Fail catastrophically against adversarial compression** (e.g., videos shared via social media at reduced quality).

### The Solution: Defense-in-Depth Pipeline

This project takes a **forensic-first** approach, modeled on how cybersecurity professionals break down attacks into layers. Instead of one "God Model," the pipeline stacks independent detection modules. A video must pass every layer to be considered authentic.

The **first and foundational layer** is the **Frequency Domain Artifact Analyzer** — a mathematically provable method that requires no GPU and produces interpretable evidence.

---

## Architecture

The pipeline follows a clean, three-stage forensic workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DeepfakePipeline (src/pipeline.py)              │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  VideoExtractor  │───▶│ FrequencyAnalyzer│───▶│ EnsembleLogic │  │
│  │                  │    │                  │    │               │  │
│  │ • OpenCV I/O     │    │ • FFT transform  │    │ • Aggregates  │  │
│  │ • MediaPipe face │    │ • Power spectrum │    │   per-frame   │  │
│  │   detection      │    │ • Azimuthal avg  │    │   scores      │  │
│  │ • frame_skip for │    │ • GAN artifact   │    │ • Weighted    │  │
│  │   performance    │    │   scoring        │    │   confidence  │  │
│  │ • 10% bbox pad   │    │                  │    │   verdict     │  │
│  └──────────────────┘    └──────────────────┘    └───────────────┘  │
│                                                          │          │
│                                              ┌───────────▼────────┐ │
│                                              │   Forensic Report  │ │
│                                              │   (JSON output)    │ │
│                                              └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| **VideoExtractor** | `backend/src/utils/video_extractor.py` | Extracts face regions from video frames using Google MediaPipe. Uses `frame_skip` to process 1-in-N frames and applies 10% bounding box padding to capture jawline blending seams. |
| **FrequencyAnalyzer** | `backend/src/detectors/frequency_analyzer.py` | Applies Fast Fourier Transform (FFT) to each face crop. Computes the 1D azimuthal power spectrum and measures high-frequency variance — the mathematical signature of GAN/diffusion upsampling. |
| **DeepfakePipeline** | `backend/src/pipeline.py` | Orchestrates the full pipeline, aggregates per-frame scores, and generates a structured JSON forensic report. |

---

## Why This Approach Is Trusted

### 1. Explainable AI — No Black Boxes

When an enterprise client asks *"Why was this video flagged?"*, the answer is mathematical, not neural. The pipeline can output the 1D power spectrum graph, showing the **exact frequency spike** caused by AI upsampling. This level of evidence is suitable for legal proceedings and audit trails.

### 2. The "Checkerboard Artifact" — The Invisible Deepfake Signature

Every AI generator (GAN, Diffusion Model, Roop, DeepFaceLab) must **upsample** a low-resolution feature map to create a high-resolution fake face. This upsampling process introduces a microscopic, periodic grid of errors — completely invisible to the human eye — known as the **checkerboard artifact**.

When we apply FFT to translate the image from the *spatial domain* (pixels) to the *frequency domain* (mathematical waves), these artifacts create **distinct, unnatural spikes in the high-frequency tail** of the power spectrum. Real human faces, filmed on real cameras, obey the natural **1/f power law** and show a smooth, monotonically decaying spectrum. A GAN cannot reproduce this.

### 3. Low Compute Overhead

The entire pipeline runs on a standard CPU. FFT analysis is **O(N log N)** — it processes a 256×256 face crop in microseconds. No GPU, no cloud credits, no 24GB VRAM required.

### 4. Adversarial Resilience via Ensemble Design

No single detector is unbeatable. This pipeline is designed to be **extended** with additional forensic modules (rPPG heartbeat, audio-visual lip-sync). If an attacker compresses a video to destroy high-frequency data and evade the FFT layer, subsequent layers still analyze biological signals and temporal consistency.

---

## Repository Structure

```
Deepfake-Detector/
├── backend/                          # Python detection engine
│   ├── requirements.txt              # Production Python dependencies
│   └── src/
│       ├── pipeline.py               # Main entry point — DeepfakePipeline
│       ├── detectors/
│       │   ├── __init__.py
│       │   └── frequency_analyzer.py # FFT-based artifact detection
│       └── utils/
│           ├── __init__.py
│           └── video_extractor.py    # MediaPipe face extraction from video
├── frontend/                         # React/TypeScript web UI
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── .env.example
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── components/
│       │   ├── DeepfakeDetector.tsx
│       │   └── Chatbot.tsx
│       └── lib/
│           └── utils.ts
└── README.md
```

---

## Backend Setup

**Python Detection Engine — Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/muhammadsohaimmuqtada/Deepfake-Detector.git
cd Deepfake-Detector/backend

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Frontend Setup

**Web UI — Prerequisites:** Node.js 18+

```bash
# From the repository root
cd frontend

# 1. Install dependencies
npm install

# 2. Copy the environment file and add your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# 3. Start the development server
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## Usage

### Command Line (Backend)

Navigate to the `backend/` directory and analyze a video file:

```bash
cd backend
python src/pipeline.py path/to/video.mp4
```

**With optional parameters:**

```bash
# Analyze every 10th frame (faster, suitable for long videos)
python src/pipeline.py path/to/video.mp4 --frame-skip 10

# Adjust the FFT artifact sensitivity threshold
python src/pipeline.py path/to/video.mp4 --frame-skip 5 --threshold 0.12
```

### Python Library

```python
import sys
sys.path.insert(0, "backend")

from src.pipeline import DeepfakePipeline

pipeline = DeepfakePipeline(frame_skip=5, freq_threshold=0.15)
report = pipeline.analyze("path/to/video.mp4")

print(f"Verdict:    {report['verdict']}")
print(f"Confidence: {report['overall_confidence']}%")
print(f"Frames analyzed: {report['frames_analyzed']}")
print(f"Frames flagged:  {report['frames_flagged']}")
```

---

## Output — Forensic Report

The pipeline returns a structured JSON object suitable for direct integration into backend APIs, audit dashboards, or logging systems.

```json
{
  "video_path": "path/to/video.mp4",
  "verdict": "FAKE",
  "is_fake": true,
  "overall_confidence": 87.34,
  "frames_analyzed": 42,
  "frames_flagged": 31,
  "fake_frame_ratio": 0.7381,
  "mean_artifact_score": 0.218456,
  "elapsed_seconds": 3.142,
  "forensic_report": [
    {
      "is_fake": true,
      "artifact_score": 0.231,
      "confidence": 77.0,
      "module": "Frequency/FFT Analyzer"
    }
  ]
}
```

| Field | Description |
|---|---|
| `verdict` | `"FAKE"`, `"REAL"`, or `"INCONCLUSIVE"` |
| `is_fake` | Boolean final determination |
| `overall_confidence` | Weighted confidence percentage (0–99%) |
| `frames_analyzed` | Number of sampled frames processed |
| `frames_flagged` | Number of frames where FFT artifacts were detected |
| `fake_frame_ratio` | Ratio of flagged frames — key ensemble signal |
| `mean_artifact_score` | Average FFT high-frequency variance across all frames |
| `elapsed_seconds` | Total wall-clock processing time |
| `forensic_report` | Per-frame breakdown for audit and explainability |

---

## Roadmap

- [x] **Layer 1:** Frequency/FFT Artifact Analyzer (complete)
- [x] **Layer 1:** MediaPipe Video Face Extractor (complete)
- [x] **Layer 1:** Ensemble Pipeline & JSON Forensic Report (complete)
- [ ] **Layer 2:** rPPG Heartbeat Analyzer — detect missing biological pulse signal
- [ ] **Layer 3:** Audio-Visual Lip-Sync Desync Analyzer (SyncNet-based)
- [ ] **Layer 4:** C2PA Content Provenance / Cryptographic Metadata Verification
- [ ] REST API wrapper (FastAPI) for backend integration
- [ ] Docker container for one-command deployment

---

<div align="center">
<sub>Built with ❤️ for forensic accuracy. Powered by mathematics, not magic.</sub>
</div>

