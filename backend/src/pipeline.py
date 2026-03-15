"""
DeepfakePipeline — main entry point for the Deepfake Detector.

Orchestrates the full forensic analysis pipeline (Defense-in-Depth):

    VideoExtractor
        ↓
    Layer 1 — FrequencyAnalyzer  (per-frame FFT artifact detection)
        ↓
    Layer 2 — RPPGAnalyzer       (temporal heartbeat / rPPG analysis)
        ↓
    Layer 3 — AudioAnalyzer      (acoustic / voice cloning detection)
        ↓
    Ensemble Scoring  →  Forensic Report (3-pillar verdict)

Usage (CLI):
    python src/pipeline.py <path_to_video> [--frame-skip N] [--threshold T]
                                           [--rppg-window W] [--fps FPS]

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

from src.detectors.audio_analyzer import AudioAnalyzer
from src.detectors.frequency_analyzer import FrequencyAnalyzer
from src.detectors.rppg_analyzer import RPPGAnalyzer
from src.utils.video_extractor import VideoExtractor

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ── Ensemble constants ────────────────────────────────────────────────────────
# Layer 1 (FFT): percentage of frames flagged before calling a video fake
FAKE_FRAME_RATIO_THRESHOLD = 0.40
MAX_CONFIDENCE_PCT = 99.0

# Layer weights in the ensemble score (must sum to 1.0 when all layers are active)
FREQ_WEIGHT = 0.40    # Layer 1 — Visual / FFT
RPPG_WEIGHT = 0.35    # Layer 2 — Biological / rPPG
AUDIO_WEIGHT = 0.25   # Layer 3 — Acoustic / Voice Cloning

# Ensemble fake threshold — weighted combined score above this → FAKE
ENSEMBLE_FAKE_THRESHOLD = 0.50

# Dynamic weight boosting: when a single layer is strongly confident about a
# FAKE result its weight is increased to reflect the severity of that attack
# vector (e.g. voice-clone attack where the face passes visual checks).
STRONG_SIGNAL_THRESHOLD = 0.70   # fake_prob above this triggers boost
AUDIO_BOOST_MAX = 0.50           # maximum extra weight added to the audio layer
RPPG_BOOST_MAX = 0.15            # maximum extra weight added to the rPPG layer

# Default number of consecutive face frames fed to the rPPG analyzer
DEFAULT_RPPG_WINDOW = 60


def _compute_boost(fake_prob: float, threshold: float, max_boost: float) -> float:
    """
    Compute a proportional weight boost for a layer whose fake probability
    exceeds *threshold*.

    :param fake_prob: The layer's fake probability in [0, 1].
    :param threshold: Minimum fake_prob required to trigger a boost.
    :param max_boost: Maximum additional weight returned when fake_prob == 1.0.
    :return: Additional weight in [0, max_boost].
    """
    if fake_prob < threshold:
        return 0.0
    scale = (fake_prob - threshold) / (1.0 - threshold + 1e-8)
    return max_boost * scale

class DeepfakePipeline:
    """
    End-to-end deepfake detection pipeline.

    Combines:
    - VideoExtractor : isolates face regions from sampled video frames.
    - FrequencyAnalyzer (Layer 1): per-frame FFT-based GAN/diffusion artifact detection.
    - RPPGAnalyzer   (Layer 2): temporal rPPG heartbeat analysis over a frame window.
    - AudioAnalyzer  (Layer 3): acoustic MFCC / spectral analysis for voice cloning detection.
    - Ensemble aggregation: fuses all three layer scores into a final 3-pillar verdict.
    """

    def __init__(
        self,
        frame_skip: int = 5,
        freq_threshold: float = 0.60,
        rppg_window: int = DEFAULT_RPPG_WINDOW,
        fps: float = 30.0,
    ):
        """
        :param frame_skip:    Analyze every Nth frame (reduces compute, preserves accuracy).
        :param freq_threshold: Spectral Energy Ratio above which a single frame is flagged (Layer 1).
                               Calibrated for the new high/low band energy ratio metric (default: 0.60).
        :param rppg_window:   Number of consecutive face frames to pass to the rPPG analyzer.
        :param fps:           Effective frame rate *after* frame-skip, used by the rPPG filter.
                              Formula: effective_fps = source_fps / frame_skip.
        """
        self.extractor = VideoExtractor(frame_skip=frame_skip)
        self.freq_analyzer = FrequencyAnalyzer(high_freq_threshold=freq_threshold)
        self.rppg_analyzer = RPPGAnalyzer(fps=fps)
        self.audio_analyzer = AudioAnalyzer()
        self.rppg_window = rppg_window

        logger.info(
            "DeepfakePipeline ready — frame_skip=%d, freq_threshold=%.3f, "
            "rppg_window=%d, effective_fps=%.1f",
            frame_skip,
            freq_threshold,
            rppg_window,
            fps,
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
        # Collect face frames for the rPPG window
        rppg_frame_buffer: list = []

        # ── Layer 3: Audio analysis (runs before the frame loop) ─────────────
        # Audio extraction is independent of frame processing and is executed
        # sequentially here so the result is ready for ensemble scoring later.
        logger.info("Pipeline — running Layer 3 (Audio Forensics) on: %s", video_path)
        audio_result = self.audio_analyzer.analyze(video_path)
        audio_conclusive = audio_result.get("verdict") != "INCONCLUSIVE"
        if not audio_conclusive:
            logger.info(
                "AudioAnalyzer returned INCONCLUSIVE (%s) — ensemble will compensate.",
                audio_result.get("reason", "unknown"),
            )

        for face_image in self.extractor.extract_faces_from_video(video_path):
            # ── Layer 1: per-frame FFT analysis ──────────────────────────────
            freq_result = self.freq_analyzer.detect_artifacts(face_image)
            frame_reports.append(freq_result)

            # ── Accumulate frames for Layer 2 ────────────────────────────────
            if len(rppg_frame_buffer) < self.rppg_window:
                rppg_frame_buffer.append(face_image)

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
                "layer1_fft_report": [],
                "layer2_rppg_report": None,
                "layer3_audio_report": audio_result,
                "message": "No faces could be detected. Video may be low quality or contain no human subjects.",
            }

        # ── Layer 1 aggregation ───────────────────────────────────────────────
        frames_analyzed = len(frame_reports)
        frames_flagged = sum(1 for r in frame_reports if r["is_fake"])
        fake_frame_ratio = frames_flagged / frames_analyzed
        mean_artifact_score = (
            sum(r["artifact_score"] for r in frame_reports) / frames_analyzed
        )

        # Layer 1 fake probability (0–1)
        freq_fake_prob = fake_frame_ratio

        # ── Layer 2: rPPG analysis on collected frame window ──────────────────
        rppg_result = self.rppg_analyzer.analyze(rppg_frame_buffer)
        rppg_conclusive = rppg_result.get("verdict") != "INCONCLUSIVE"

        if rppg_conclusive:
            rppg_conf = float(rppg_result["confidence"]) / 100.0
            rppg_fake_prob = rppg_conf if rppg_result["is_fake"] else (1.0 - rppg_conf)
        else:
            rppg_fake_prob = 0.5
            logger.info("rPPG returned INCONCLUSIVE — ensemble compensates on remaining layers.")

        # ── Layer 3: Audio fake probability ───────────────────────────────────
        if audio_conclusive:
            audio_conf = float(audio_result["confidence"]) / 100.0
            audio_fake_prob = audio_conf if audio_result["is_fake"] else (1.0 - audio_conf)
        else:
            audio_fake_prob = 0.5

        # ── Dynamic weight boosting (Weighted Ensemble) ───────────────────────
        # If a single layer is strongly confident about a FAKE verdict, its
        # weight is boosted to reflect the severity of that specific attack
        # vector.  The canonical example is a *voice-clone attack*: the face
        # passes the visual and biological checks, but the audio is definitively
        # synthetic.  Without boosting, the audio layer (base weight 0.25) can
        # be drowned out by two "REAL" verdicts.  With boosting, a very strong
        # audio FAKE signal raises the audio weight up to (0.25 + 0.50 = 0.75)
        # before re-normalisation, ensuring the voice clone is not missed.
        audio_boost = 0.0
        if audio_conclusive and audio_fake_prob >= STRONG_SIGNAL_THRESHOLD:
            audio_boost = _compute_boost(audio_fake_prob, STRONG_SIGNAL_THRESHOLD, AUDIO_BOOST_MAX)
            logger.info(
                "Ensemble — audio boost applied: +%.3f (audio_fake_prob=%.3f)",
                audio_boost,
                audio_fake_prob,
            )

        rppg_boost = 0.0
        if rppg_conclusive and rppg_fake_prob >= STRONG_SIGNAL_THRESHOLD:
            rppg_boost = _compute_boost(rppg_fake_prob, STRONG_SIGNAL_THRESHOLD, RPPG_BOOST_MAX)
            logger.info(
                "Ensemble — rPPG boost applied: +%.3f (rppg_fake_prob=%.3f)",
                rppg_boost,
                rppg_fake_prob,
            )

        # ── Ensemble scoring (3-pillar) ────────────────────────────────────────
        # Re-normalise weights based on which layers returned conclusive results
        # and incorporate any dynamic boosts.
        active_weights: dict[str, float] = {
            "freq": FREQ_WEIGHT,
            "rppg": (RPPG_WEIGHT + rppg_boost) if rppg_conclusive else 0.0,
            "audio": (AUDIO_WEIGHT + audio_boost) if audio_conclusive else 0.0,
        }
        total_weight = sum(active_weights.values())
        if total_weight == 0.0:
            total_weight = 1.0  # fallback (should never happen)

        ensemble_fake_prob = (
            active_weights["freq"] * freq_fake_prob
            + active_weights["rppg"] * rppg_fake_prob
            + active_weights["audio"] * audio_fake_prob
        ) / total_weight

        is_fake = ensemble_fake_prob >= ENSEMBLE_FAKE_THRESHOLD
        verdict = "FAKE" if is_fake else "REAL"

        # Overall confidence — how far from the decision boundary
        if is_fake:
            overall_confidence = round(
                min(ensemble_fake_prob * MAX_CONFIDENCE_PCT, MAX_CONFIDENCE_PCT), 2
            )
        else:
            overall_confidence = round(
                min((1.0 - ensemble_fake_prob) * MAX_CONFIDENCE_PCT, MAX_CONFIDENCE_PCT), 2
            )

        report = {
            "video_path": video_path,
            "verdict": verdict,
            "is_fake": is_fake,
            "overall_confidence": overall_confidence,
            # ── Layer 1 (Visual / FFT) summary ──
            "frames_analyzed": frames_analyzed,
            "frames_flagged": frames_flagged,
            "fake_frame_ratio": round(fake_frame_ratio, 4),
            "mean_artifact_score": round(mean_artifact_score, 6),
            # ── Ensemble metrics ──
            "ensemble_fake_probability": round(ensemble_fake_prob, 4),
            "layer_weights": {
                "visual_fft": round(active_weights["freq"] / total_weight, 4),
                "biological_rppg": round(active_weights["rppg"] / total_weight, 4),
                "acoustic_audio": round(active_weights["audio"] / total_weight, 4),
            },
            "elapsed_seconds": round(elapsed, 3),
            # ── Detailed per-layer reports ──
            "layer1_fft_report": frame_reports,
            "layer2_rppg_report": rppg_result,
            "layer3_audio_report": audio_result,
        }

        logger.info(
            "Analysis complete — verdict=%s, confidence=%.2f%%, ensemble_prob=%.4f, "
            "frames=%d, flagged=%d, rppg_snr=%.3f, audio_score=%.4f, elapsed=%.3fs",
            verdict,
            overall_confidence,
            ensemble_fake_prob,
            frames_analyzed,
            frames_flagged,
            rppg_result.get("rppg_snr", 0.0),
            audio_result.get("artifact_score", 0.0),
            elapsed,
        )
        return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake-detector",
        description=(
            "Production-ready Deepfake Detection Pipeline — "
            "Layer 1: FFT frequency analysis | Layer 2: rPPG heartbeat analysis | "
            "Layer 3: Audio forensics / voice cloning detection"
        ),
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
        default=0.60,
        metavar="T",
        help="Spectral Energy Ratio threshold per frame (default: 0.60). Higher = more strict.",
    )
    parser.add_argument(
        "--rppg-window",
        type=int,
        default=DEFAULT_RPPG_WINDOW,
        metavar="W",
        help=f"Number of face frames to use for rPPG analysis (default: {DEFAULT_RPPG_WINDOW}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Effective frames-per-second after frame-skip (default: 30.0). "
             "Set to source_fps / frame_skip for accurate rPPG filtering.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    pipeline = DeepfakePipeline(
        frame_skip=args.frame_skip,
        freq_threshold=args.threshold,
        rppg_window=args.rppg_window,
        fps=args.fps,
    )

    try:
        result = pipeline.analyze(args.video_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pipeline error: %s", exc)
        sys.exit(1)

    print(json.dumps(result, indent=2))
