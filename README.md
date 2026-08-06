# Deepfake Detector

A local experimental media-forensics pipeline for exploring non-neural deepfake detection signals across video and audio.

The project combines three heuristic analysis paths:

- frequency-domain analysis of sampled face regions
- remote photoplethysmography (rPPG) signal analysis
- acoustic feature analysis using MFCC-derived measurements

A FastAPI backend exposes the analysis pipeline and a React/Vite frontend provides an interface for submitting media and reviewing the resulting signals.

> **Research / portfolio prototype:** this project is not a validated forensic product and its output should not be treated as proof that media is authentic or synthetic. Deepfake detection is an adversarial and rapidly changing problem; heuristic signals can fail under compression, editing, lighting changes, capture conditions, or newer generation methods.

## Architecture

```text
Video
  |
  +-- face extraction
  |
  +-- frequency analysis (FFT)
  |
  +-- rPPG analysis
  |
  +-- audio extraction / MFCC analysis
  |
  +-- pipeline aggregation
  |
  +-- FastAPI response
          |
          +-- React/Vite interface
```

## Repository structure

```text
Deepfake-Detector/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── src/
│       ├── pipeline.py
│       ├── detectors/
│       │   ├── frequency_analyzer.py
│       │   ├── rppg_analyzer.py
│       │   └── audio_analyzer.py
│       └── utils/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
└── README.md
```

## Analysis paths

### Frequency-domain analysis

The visual path samples face regions and evaluates frequency-domain characteristics using FFT-based measurements. The intent is to explore whether spectral artifacts provide useful evidence when combined with other signals rather than to treat them as a universal deepfake signature.

### rPPG analysis

The biological path examines temporal color variation in detected face regions and derives signal-quality measurements associated with remote photoplethysmography. Results are sensitive to lighting, motion, compression, skin visibility, frame rate, and capture quality.

### Acoustic analysis

The audio path extracts speech-related features and evaluates MFCC-derived measurements, including temporal changes. These measurements are experimental indicators and are not a substitute for a trained, independently evaluated voice-clone detector.

## Getting started

### Requirements

- Python 3.10+
- Node.js 18+

### Backend

```bash
git clone https://github.com/muhammadsohaimmuqtada/Deepfake-Detector.git
cd Deepfake-Detector

python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn main:app --reload
```

The API is available locally at `http://localhost:8000` with interactive documentation at `/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API

`POST /api/analyze` accepts a video upload and returns the pipeline's per-signal measurements and aggregate result as JSON.

## What this project demonstrates

- video preprocessing and face-region extraction
- signal processing with NumPy/SciPy/OpenCV
- experimental rPPG analysis
- audio feature extraction
- aggregation of multiple forensic indicators
- FastAPI service design
- React/Vite frontend integration

## Limitations

This repository does not claim production detection accuracy. Before any real forensic use, the pipeline would need a documented evaluation methodology, representative real/deepfake datasets, calibration, false-positive/false-negative analysis, robustness testing against transformations, and independent validation.

Useful future work includes dataset-backed benchmarking, calibration by media quality, provenance checks such as C2PA, audio/video synchronization analysis, and reproducible evaluation scripts.

## License

MIT License.