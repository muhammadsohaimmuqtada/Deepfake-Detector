"""
DeepfakePipeline — main entry point for the Deepfake Detector.

Orchestrates the full forensic analysis pipeline:
    VideoExtractor  →  FrequencyAnalyzer  →  Ensemble Scoring  →  Forensic Report

Usage (CLI):
    python src/pipeline.py <path_to_video> [--frame-skip N] [--threshold T]

Usage (library):
    from src.pipeline import DeepfakePipeline
    pipeline = DeepfakePipeline()
    report = pipeline.analyze("path/to/video.mp4")
    print(report)
"""

import argparse
import json
import logging
import sys
import time
from typing import Any

from src.detectors.frequency_analyzer import FrequencyAnalyzer
from src.utils.video_extractor import VideoExtractor

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Ensemble decision threshold — percentage of frames flagged to call a video fake
FAKE_FRAME_RATIO_THRESHOLD = 0.40
MAX_CONFIDENCE_PCT = 99.0


class DeepfakePipeline:
    """
    End-to-end deepfake detection pipeline.

    Combines:
    - VideoExtractor: isolates face regions from sampled video frames.
    - FrequencyAnalyzer: detects GAN/diffusion upsampling artifacts via FFT.
    - Ensemble aggregation: aggregates per-frame scores to produce a final verdict.
    """

    def __init__(
        self,
        frame_skip: int = 5,
        freq_threshold: float = 0.15,
    ):
        """
        :param frame_skip: Analyze every Nth frame (reduces compute, preserves accuracy).
        :param freq_threshold: FFT artifact score above which a single frame is flagged.
        """
        self.extractor = VideoExtractor(frame_skip=frame_skip)
        self.analyzer = FrequencyAnalyzer(high_freq_threshold=freq_threshold)
        logger.info(
            "DeepfakePipeline ready — frame_skip=%d, freq_threshold=%.3f",
            frame_skip,
            freq_threshold,
        )

    def analyze(self, video_path: str) -> dict[str, Any]:
        """
        Run the full forensic analysis on a video file.

        :param video_path: Path to the video file.
        :return: A JSON-serializable forensic report dictionary.
        :raises FileNotFoundError: If the video file does not exist.
        :raises RuntimeError: If no faces are detected in the video.
        """
        logger.info("Pipeline started — analyzing: %s", video_path)
        start_time = time.monotonic()

        frame_reports: list[dict[str, Any]] = []

        for face_image in self.extractor.extract_faces_from_video(video_path):
            result = self.analyzer.detect_artifacts(face_image)
            frame_reports.append(result)

        elapsed = time.monotonic() - start_time

        if not frame_reports:
            logger.warning("No faces detected in video: %s", video_path)
            return {
                "video_path": video_path,
                "verdict": "INCONCLUSIVE",
                "is_fake": False,
                "overall_confidence": 0.0,
                "frames_analyzed": 0,
                "frames_flagged": 0,
                "fake_frame_ratio": 0.0,
                "mean_artifact_score": 0.0,
                "elapsed_seconds": round(elapsed, 3),
                "forensic_report": [],
                "message": "No faces could be detected. Video may be low quality or contain no human subjects.",
            }

        frames_analyzed = len(frame_reports)
        frames_flagged = sum(1 for r in frame_reports if r["is_fake"])
        fake_frame_ratio = frames_flagged / frames_analyzed
        mean_artifact_score = sum(r["artifact_score"] for r in frame_reports) / frames_analyzed
        mean_confidence = sum(r["confidence"] for r in frame_reports) / frames_analyzed

        is_fake = fake_frame_ratio >= FAKE_FRAME_RATIO_THRESHOLD
        verdict = "FAKE" if is_fake else "REAL"

        # Overall confidence — weighted by fake ratio when fake, inverse when real
        if is_fake:
            overall_confidence = round(min(fake_frame_ratio * mean_confidence, MAX_CONFIDENCE_PCT), 2)
        else:
            overall_confidence = round(min((1.0 - fake_frame_ratio) * mean_confidence, MAX_CONFIDENCE_PCT), 2)

        report = {
            "video_path": video_path,
            "verdict": verdict,
            "is_fake": is_fake,
            "overall_confidence": overall_confidence,
            "frames_analyzed": frames_analyzed,
            "frames_flagged": frames_flagged,
            "fake_frame_ratio": round(fake_frame_ratio, 4),
            "mean_artifact_score": round(mean_artifact_score, 6),
            "elapsed_seconds": round(elapsed, 3),
            "forensic_report": frame_reports,
        }

        logger.info(
            "Analysis complete — verdict=%s, confidence=%.2f%%, frames_analyzed=%d, "
            "frames_flagged=%d, elapsed=%.3fs",
            verdict,
            overall_confidence,
            frames_analyzed,
            frames_flagged,
            elapsed,
        )
        return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake-detector",
        description="Production-ready Deepfake Detection Pipeline using FFT frequency analysis.",
    )
    parser.add_argument("video_path", help="Path to the video file to analyze.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        metavar="N",
        help="Analyze every Nth frame (default: 5). Higher = faster, lower = more thorough.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        metavar="T",
        help="FFT artifact score threshold per frame (default: 0.15).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    pipeline = DeepfakePipeline(
        frame_skip=args.frame_skip,
        freq_threshold=args.threshold,
    )

    try:
        result = pipeline.analyze(args.video_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pipeline error: %s", exc)
        sys.exit(1)

    print(json.dumps(result, indent=2))
