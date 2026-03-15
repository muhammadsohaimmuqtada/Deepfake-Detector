<div align="center">

<h1>🛡️ Deepfake Detector</h1>

<p><strong>An enterprise-grade, multi-layer forensic engine for detecting AI-generated synthetic media.</strong><br>
No black boxes. No GPU required. Fully explainable math — from FFT to rPPG to acoustic forensics.</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19%2B-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6%2B-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4%2B-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B.svg)](LICENSE)
[![Offline Capable](https://img.shields.io/badge/Offline-100%25-brightgreen)](https://github.com/muhammadsohaimmuqtada/Deepfake-Detector)

</div>

---

## 📋 Table of Contents

- [Why This Exists](#-why-this-exists)
- [The Defense-in-Depth Architecture](#-the-defense-in-depth-architecture)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Forensic Report Output](#-forensic-report-output)
- [Roadmap](#-roadmap)

---

## 🎯 Why This Exists

AI-generated "deepfake" videos are one of the most critical threats to information integrity in the digital age. Public figures can be impersonated. Evidence can be fabricated. Legacy detection systems fail because they:

- **Act as neural black boxes** — they produce a verdict without evidence, making them inadmissible for audit or legal review.
- **Require expensive GPU infrastructure** — cloud costs and hardware barriers put them out of reach for most organizations.
- **Break under social-media compression** — a single re-upload can wipe the artifacts a single-model detector was trained on.

**Deepfake Detector** takes a different approach: **forensic-first, math-based, Defense-in-Depth**. Instead of one overconfident neural network, it runs three independent, physics-grounded interrogation layers against every uploaded video. A synthetic video must fool all three simultaneously to evade detection.

---

## 🏗️ The Defense-in-Depth Architecture

The engine interrogates every video across three independent forensic vectors. All three must pass for a video to be declared authentic.

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       DEEPFAKE DETECTOR ENGINE                          │
  │                                                                         │
  │   VIDEO IN ──▶  Face Extractor (MediaPipe)                              │
  │                        │                                                │
  │         ┌──────────────┼──────────────┐                                 │
  │         ▼              ▼              ▼                                 │
  │   ┌───────────┐  ┌───────────┐  ┌───────────┐                          │
  │   │  LAYER 1  │  │  LAYER 2  │  │  LAYER 3  │                          │
  │   │ Visual    │  │Biological │  │  Acoustic │                          │
  │   │  (FFT)    │  │  (rPPG)   │  │  (MFCC)   │                          │
  │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                          │
  │         └──────────────┴──────────────┘                                 │
  │                         │                                               │
  │                ┌────────▼────────┐                                      │
  │                │  Weighted Threat │                                      │
  │                │     Matrix       │                                      │
  │                └────────┬────────┘                                      │
  │                         ▼                                               │
  │               ┌──────────────────┐                                      │
  │               │  Forensic Report │  (JSON — API or CLI)                 │
  │               └──────────────────┘                                      │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Layer 1 · Visual Math (FFT) — *Detecting Invisible AI Upsampling Artifacts*

Every AI face generator (GAN, Diffusion Model, Roop, DeepFaceLab) must **upsample** a low-resolution feature map into a high-resolution face. This upsampling introduces a microscopic, periodic grid of pixel errors — completely invisible to the human eye — known as the **checkerboard artifact**.

The engine applies a **Fast Fourier Transform (FFT)** to convert each face crop from pixel space into frequency space. It then computes the **Spectral Energy Ratio** between low and high frequency bands. Real human faces obey the natural **1/f power law** (a smooth, decaying spectrum). AI-generated faces break this law, creating unnatural spikes in the high-frequency tail. The math catches what human eyes never could.

### Layer 2 · Biological Pulse (rPPG) — *Detecting the Presence of a Real Human Heartbeat*

Every living human face has a **remote photoplethysmography (rPPG) signal** — a subtle, cyclic variation in skin color (in the green channel) caused by blood flow with each heartbeat, typically 45–180 BPM.

The engine analyzes micro-color oscillations frame-by-frame and computes the **Signal-to-Noise Ratio (SNR)** of the extracted pulse waveform. A real face produces a clear, dominant pulse spike above the noise floor. An AI-generated face — which is a mathematically constructed texture — has no biological pulse. It produces only chaotic static. If the SNR falls below the threshold, the subject is declared biologically absent.

### Layer 3 · Acoustic Forensics (MFCC) — *Detecting Robotic Micro-Jitters in AI Voice Clones*

State-of-the-art voice cloners (e.g., ElevenLabs) can closely replicate base vocal frequencies, but they struggle to reproduce one thing: **vocal tract acceleration over time**.

The engine strips the audio from the video using `moviepy`, then extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** — the same math used in professional speech recognition. Crucially, it also computes the **Delta and Delta-Delta (acceleration) coefficients**. Human vocal cords are naturally imperfect; the micro-jitter and shimmer in real speech follow chaotic organic patterns. AI voices are too mathematically smooth. The MFCC acceleration analysis catches the "robotic" lack of natural vocal dynamics.

### The Weighted Threat Matrix

The three layer scores are not simply averaged. The pipeline applies a **weighted threat matrix**. If the face passes visual and biological analysis but the audio fails, the system recognizes this as a **Voice Clone Attack** — a distinct and serious threat vector — and scores it accordingly. A single decisive layer failure is enough to trigger a high-confidence `SYNTHETIC THREAT` verdict.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Face Detection** | [MediaPipe](https://mediapipe.dev) | Real-time face landmark detection and bounding box extraction |
| **Visual Analysis** | [OpenCV](https://opencv.org) + [NumPy](https://numpy.org) | Frame I/O, face cropping, FFT/spectral analysis |
| **Biological Analysis** | [SciPy](https://scipy.org) | Signal processing for rPPG heartbeat SNR calculation |
| **Acoustic Analysis** | [Librosa](https://librosa.org) + [MoviePy](https://zulko.github.io/moviepy/) | Audio extraction, MFCC / Delta-Delta coefficient analysis |
| **API Server** | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | High-performance async REST API |
| **Frontend** | [React 19](https://react.dev) + [Vite 6](https://vitejs.dev) | Component-based web UI with hot module replacement |
| **Styling** | [Tailwind CSS 4](https://tailwindcss.com) | Utility-first dark-mode cybersecurity aesthetic |
| **Language** | Python 3.10+ / TypeScript 5.8+ | Type-safe, production-grade across the full stack |

---

## 📁 Repository Structure

```
Deepfake-Detector/
├── backend/                          # Python forensic engine
│   ├── main.py                       # FastAPI server — POST /api/analyze
│   ├── requirements.txt              # Python dependencies
│   └── src/
│       ├── pipeline.py               # Core orchestrator — DeepfakePipeline
│       ├── detectors/
│       │   └── frequency_analyzer.py # Layer 1: FFT Spectral Energy Ratio
│       └── utils/
│           └── video_extractor.py    # MediaPipe face extractor
├── frontend/                         # React/Vite web dashboard
│   ├── package.json
│   ├── vite.config.ts
│   ├── .env.example                  # Environment variable template
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── DeepfakeDetector.tsx  # Upload zone + forensic report UI
│       │   └── Chatbot.tsx           # AI assistant interface
│       └── lib/
│           └── utils.ts
└── README.md
```

---

## 🚀 Getting Started

Run both services simultaneously in two terminal windows.

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**

---

### Terminal 1 — Start the Forensic API (Backend)

```bash
# Clone the repository
git clone https://github.com/muhammadsohaimmuqtada/Deepfake-Detector.git
cd Deepfake-Detector

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server
cd backend
uvicorn main:app --reload
```

> The API will be live at **`http://localhost:8000`**  
> Interactive Swagger docs available at **`http://localhost:8000/docs`**

---

### Terminal 2 — Start the Web Dashboard (Frontend)

```bash
# From the repository root
cd frontend

# Install dependencies
npm install

# (Optional) Configure environment — copy the example and set your Gemini API key
cp .env.example .env

# Start the development server
npm run dev
```

> The dashboard will be live at **`http://localhost:3000`**

---

### Quick Test

Once both services are running:

1. Open **`http://localhost:3000`** in your browser.
2. Drag and drop an `.mp4` video file into the upload zone.
3. Watch the terminal-style progress indicator run through all three forensic layers.
4. Review the **Forensic Report** — three independent layer verdicts plus a final confidence score.

---

## 📡 API Reference

### `POST /api/analyze`

Submits a video file for multi-layer forensic analysis.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `File` | The `.mp4` video to analyze |

**Response:** `application/json`

```json
{
  "verdict": "FAKE",
  "is_fake": true,
  "overall_confidence": 94.7,
  "visual_artifacts_detected": true,
  "biological_pulse_detected": false,
  "synthetic_audio_detected": true,
  "frames_analyzed": 42,
  "frames_flagged": 36,
  "fake_frame_ratio": 0.857,
  "mean_artifact_score": 0.231,
  "elapsed_seconds": 4.81,
  "forensic_report": [
    {
      "layer": "FFT / Visual",
      "is_fake": true,
      "artifact_score": 0.231,
      "confidence": 91.0
    },
    {
      "layer": "rPPG / Biological",
      "is_fake": true,
      "pulse_snr": 1.2,
      "confidence": 88.0
    },
    {
      "layer": "MFCC / Acoustic",
      "is_fake": true,
      "delta_delta_variance": 0.004,
      "confidence": 97.5
    }
  ]
}
```

---

## 📊 Forensic Report Output

| Field | Description |
|---|---|
| `verdict` | `"FAKE"`, `"REAL"`, or `"INCONCLUSIVE"` |
| `is_fake` | Boolean final determination |
| `overall_confidence` | Weighted confidence score (0–100%) |
| `visual_artifacts_detected` | Layer 1 result — FFT checkerboard artifact detection |
| `biological_pulse_detected` | Layer 2 result — rPPG heartbeat SNR check |
| `synthetic_audio_detected` | Layer 3 result — MFCC Delta-Delta vocal analysis |
| `frames_analyzed` | Total sampled frames processed |
| `frames_flagged` | Frames where visual artifacts exceeded threshold |
| `fake_frame_ratio` | Proportion of flagged frames (key ensemble signal) |
| `mean_artifact_score` | Mean FFT high-frequency variance across all frames |
| `elapsed_seconds` | Total wall-clock analysis time |
| `forensic_report` | Per-layer breakdown for audit trails and explainability |

---

## 🗺️ Roadmap

- [x] **Layer 1:** FFT Spectral Energy Ratio — visual artifact analysis
- [x] **Layer 2:** rPPG Heartbeat SNR — biological pulse verification
- [x] **Layer 3:** MFCC Delta-Delta — acoustic voice clone detection
- [x] FastAPI REST server with Swagger documentation
- [x] React/Vite dark-mode forensic dashboard
- [x] Weighted Threat Matrix ensemble scoring
- [ ] Docker Compose for one-command deployment
- [ ] Layer 4: Audio-visual lip-sync desync analysis (SyncNet-based)
- [ ] Layer 5: C2PA cryptographic content provenance verification
- [ ] Batch processing endpoint for video archives
- [ ] Exportable PDF forensic report for legal proceedings

---

<div align="center">

**Built for the truth. Powered by mathematics, not magic.**

<sub>© 2024 Muhammad Sohaim Muqtada · MIT License</sub>

</div>

