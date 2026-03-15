"""
FastAPI backend for the Deepfake Detector.

Exposes a single endpoint:
    POST /api/analyze  — accepts a video file upload and returns a JSON forensic report.

Run with:
    uvicorn main:app --reload
"""

import os
import sys
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from the src package when running from the backend/ directory
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import DeepfakePipeline  # noqa: E402

app = FastAPI(
    title="Deepfake Detector API",
    description="Production-ready deepfake detection pipeline using FFT frequency analysis.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (Vite default: 5173) and any localhost
# origin so the frontend can communicate with this backend.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the pipeline once at startup (avoids re-loading MediaPipe on every request)
_pipeline = DeepfakePipeline()


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accept a video file upload, run it through the DeepfakePipeline, and return
    the JSON forensic report.

    - **file**: The video file to analyze (mp4, avi, mov, etc.)
    """
    # Save the uploaded file to a temporary location
    suffix = os.path.splitext(file.filename or "upload")[1] or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await file.read()
        with os.fdopen(tmp_fd, "wb") as tmp:
            tmp.write(content)

        report = _pipeline.analyze(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return report
