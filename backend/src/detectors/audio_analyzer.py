"""
AudioAnalyzer — Layer 3 Forensic Module: Audio Forensics / Voice Cloning Detector.

Core Principle (The AI Flaw):
    Human vocal cords produce sound with natural micro-fluctuations: slight,
    irregular variations in fundamental frequency (jitter) and amplitude (shimmer),
    as well as natural breath noise and room reverberation.  AI voice cloning
    systems (e.g., ElevenLabs, Tortoise-TTS, RVC) synthesise audio by predicting
    a statistically average waveform, which is mathematically too smooth.  This
    manifests as:
        1. Abnormally low variance in the high-frequency MFCC coefficients.
        2. Unnatural spectral flatness in the upper Mel bands (lack of breathiness).
        3. Low zero-crossing rate variance relative to the overall energy.

Algorithm:
    1. Extract the audio track from the video file using FFmpeg (subprocess).
       Falls back to moviepy if FFmpeg is not on PATH.
    2. Load the raw waveform with librosa.
    3. Compute MFCCs (first 20 coefficients) over the full signal.
    4. Compute the variance of each MFCC coefficient across time frames.
    5. Derive a synthetic artifact score from the mean and spread of variances:
       very low variance → AI-smoothed; high/erratic variance → human noise.
    6. Compute the spectral flatness and zero-crossing rate as corroborating signals.
    7. Return a structured forensic result compatible with the pipeline ensemble.

Usage:
    from src.detectors.audio_analyzer import AudioAnalyzer
    analyzer = AudioAnalyzer()
    result = analyzer.analyze("/path/to/video.mp4")
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Detection constants ───────────────────────────────────────────────────────
N_MFCC = 20                   # Number of MFCC coefficients to extract
SAMPLE_RATE = 22050           # Target sample rate for librosa (Hz)
# A synthetic voice has artificially low MFCC variance.
# Empirically calibrated: human speech MFCC variance usually exceeds this value.
MFCC_VARIANCE_FAKE_THRESHOLD = 30.0
# Spectral flatness close to 1.0 → noise/synthetic; close to 0.0 → tonal/human
SPECTRAL_FLATNESS_FAKE_THRESHOLD = 0.15
# Weight for combining MFCC and spectral-flatness signals into the artifact score
MFCC_WEIGHT = 0.70
FLATNESS_WEIGHT = 0.30


class AudioAnalyzer:
    """
    Audio forensics analyzer for voice cloning / synthetic speech detection.

    Extracts the audio track from a video file and applies acoustic signal
    analysis (MFCCs, spectral flatness, zero-crossing rate) to detect the
    characteristic mathematical smoothness of AI-generated voices.
    """

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        sample_rate: int = SAMPLE_RATE,
        mfcc_var_threshold: float = MFCC_VARIANCE_FAKE_THRESHOLD,
        spectral_flatness_threshold: float = SPECTRAL_FLATNESS_FAKE_THRESHOLD,
    ) -> None:
        """
        :param n_mfcc: Number of MFCC coefficients to compute (default: 20).
        :param sample_rate: Target audio sample rate in Hz (default: 22 050).
        :param mfcc_var_threshold: Mean MFCC variance below this → synthetic flag.
        :param spectral_flatness_threshold: Mean spectral flatness above this → noisy/synthetic.
        """
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.mfcc_var_threshold = mfcc_var_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        logger.info(
            "AudioAnalyzer ready — n_mfcc=%d, sr=%d, mfcc_var_threshold=%.2f, "
            "flatness_threshold=%.4f",
            n_mfcc,
            sample_rate,
            mfcc_var_threshold,
            spectral_flatness_threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, video_path: str) -> dict[str, Any]:
        """
        Run the audio forensics analysis on a video file.

        If the video contains no audio track, or if audio extraction fails for
        any reason, returns an INCONCLUSIVE result so the pipeline can continue
        with the visual layers unaffected.

        :param video_path: Absolute or relative path to the video file.
        :return: Forensic result dictionary compatible with pipeline ensemble.
        """
        logger.info("AudioAnalyzer — starting analysis for: %s", video_path)

        # ── Step 1: Extract audio to a temporary WAV file ─────────────────────
        audio_path: str | None = None
        tmp_dir = tempfile.mkdtemp(prefix="deepfake_audio_")
        try:
            audio_path = os.path.join(tmp_dir, "extracted_audio.wav")
            extraction_ok = self._extract_audio(video_path, audio_path)

            if not extraction_ok:
                logger.warning(
                    "AudioAnalyzer — no audio track found or extraction failed: %s",
                    video_path,
                )
                return self._inconclusive_result(reason="no_audio_track")

            # ── Step 2: Load waveform ─────────────────────────────────────────
            waveform, sr = self._load_audio(audio_path)
            if waveform is None or len(waveform) == 0:
                logger.warning("AudioAnalyzer — loaded empty waveform from: %s", audio_path)
                return self._inconclusive_result(reason="empty_waveform")

            # ── Step 3: Compute acoustic features ────────────────────────────
            mfcc_result = self._compute_mfcc_features(waveform, sr)
            flatness_result = self._compute_spectral_flatness(waveform, sr)
            zcr_result = self._compute_zcr_features(waveform)

            # ── Step 4: Fuse features → artifact score ────────────────────────
            artifact_score, is_fake, confidence = self._compute_verdict(
                mfcc_result, flatness_result, zcr_result
            )

            logger.info(
                "AudioAnalyzer — is_fake=%s, artifact_score=%.4f, confidence=%.1f%%",
                is_fake,
                artifact_score,
                confidence,
            )

            return {
                "module": "Audio Forensics Analyzer",
                "is_fake": is_fake,
                "artifact_score": round(float(artifact_score), 6),
                "confidence": round(float(confidence), 2),
                "mfcc_mean_variance": round(float(mfcc_result["mean_variance"]), 4),
                "mfcc_variance_threshold": self.mfcc_var_threshold,
                "spectral_flatness_mean": round(float(flatness_result["mean_flatness"]), 6),
                "spectral_flatness_threshold": self.spectral_flatness_threshold,
                "zcr_mean": round(float(zcr_result["zcr_mean"]), 6),
                "zcr_variance": round(float(zcr_result["zcr_variance"]), 8),
                "audio_duration_seconds": round(float(len(waveform) / sr), 3),
                "sample_rate": sr,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "AudioAnalyzer — unexpected error during analysis: %s", exc, exc_info=True
            )
            return self._inconclusive_result(reason=f"analysis_error: {type(exc).__name__}")

        finally:
            # Always clean up the temporary audio file
            self._cleanup_temp(tmp_dir)

    # ── Audio extraction ──────────────────────────────────────────────────────

    def _extract_audio(self, video_path: str, output_wav: str) -> bool:
        """
        Extract the audio track from a video file to a WAV file.

        Tries FFmpeg first (fast, no Python overhead).  Falls back to moviepy
        if FFmpeg is not available on PATH.

        :param video_path: Source video file path.
        :param output_wav: Destination WAV file path.
        :return: True if a non-empty audio file was produced, False otherwise.
        """
        if self._extract_with_ffmpeg(video_path, output_wav):
            return True
        logger.info("AudioAnalyzer — FFmpeg unavailable or failed; trying moviepy fallback.")
        return self._extract_with_moviepy(video_path, output_wav)

    @staticmethod
    def _extract_with_ffmpeg(video_path: str, output_wav: str) -> bool:
        """Use FFmpeg subprocess to extract audio (preferred — zero import overhead)."""
        if not shutil.which("ffmpeg"):
            logger.debug("AudioAnalyzer — FFmpeg not found on PATH.")
            return False
        try:
            cmd = [
                "ffmpeg",
                "-y",                      # Overwrite output without prompt
                "-i", video_path,          # Input video
                "-vn",                     # Skip video stream
                "-acodec", "pcm_s16le",    # PCM 16-bit little-endian WAV
                "-ar", str(SAMPLE_RATE),   # Resample to target rate
                "-ac", "1",               # Mono channel
                output_wav,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                # FFmpeg writes error info to stderr
                stderr_snippet = result.stderr[-400:].decode("utf-8", errors="replace")
                logger.warning(
                    "AudioAnalyzer — FFmpeg exited with code %d: %s",
                    result.returncode,
                    stderr_snippet,
                )
                return False
            # Confirm the output file is non-empty
            if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
                logger.warning("AudioAnalyzer — FFmpeg produced empty output file.")
                return False
            logger.info("AudioAnalyzer — FFmpeg audio extraction succeeded.")
            return True
        except subprocess.TimeoutExpired:
            logger.error("AudioAnalyzer — FFmpeg timed out during audio extraction.")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — FFmpeg extraction error: %s", exc)
            return False

    @staticmethod
    def _extract_with_moviepy(video_path: str, output_wav: str) -> bool:
        """Use moviepy as a fallback audio extractor."""
        try:
            from moviepy.editor import VideoFileClip  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "AudioAnalyzer — moviepy is not installed; cannot extract audio."
            )
            return False
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    logger.info("AudioAnalyzer — moviepy: video has no audio track.")
                    return False
                clip.audio.write_audiofile(
                    output_wav,
                    fps=SAMPLE_RATE,
                    nbytes=2,    # 16-bit
                    ffmpeg_params=["-ac", "1"],  # Mono
                    logger=None,
                )
            if not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
                logger.warning("AudioAnalyzer — moviepy produced empty output file.")
                return False
            logger.info("AudioAnalyzer — moviepy audio extraction succeeded.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — moviepy extraction error: %s", exc)
            return False

    # ── Waveform loading ──────────────────────────────────────────────────────

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray | None, int]:
        """
        Load a WAV file into a float32 NumPy array using librosa.

        :return: (waveform, sample_rate) or (None, 0) on failure.
        """
        try:
            import librosa  # noqa: PLC0415

            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(
                "AudioAnalyzer — loaded audio: %.2f seconds at %d Hz (%d samples).",
                len(waveform) / sr,
                sr,
                len(waveform),
            )
            return waveform, sr
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — librosa load failed: %s", exc)
            return None, 0

    # ── Feature extraction ────────────────────────────────────────────────────

    def _compute_mfcc_features(
        self, waveform: np.ndarray, sr: int
    ) -> dict[str, float]:
        """
        Compute MFCCs and return variance statistics.

        AI voices are too smooth → the variance of each MFCC coefficient
        across time is lower than for natural human speech.

        :return: {'mean_variance': float, 'std_variance': float}
        """
        try:
            import librosa  # noqa: PLC0415

            mfccs = librosa.feature.mfcc(
                y=waveform, sr=sr, n_mfcc=self.n_mfcc
            )  # shape: (n_mfcc, T)
            # Variance of each coefficient across time frames
            coeff_variances = np.var(mfccs, axis=1)
            mean_var = float(np.mean(coeff_variances))
            std_var = float(np.std(coeff_variances))
            logger.debug(
                "AudioAnalyzer — MFCC variance: mean=%.4f, std=%.4f", mean_var, std_var
            )
            return {"mean_variance": mean_var, "std_variance": std_var}
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — MFCC computation failed: %s", exc)
            return {"mean_variance": self.mfcc_var_threshold, "std_variance": 0.0}

    @staticmethod
    def _compute_spectral_flatness(
        waveform: np.ndarray, sr: int
    ) -> dict[str, float]:
        """
        Compute spectral flatness (Wiener entropy) across the signal.

        Spectral flatness near 1 → white-noise-like / synthetic smoothing.
        Near 0 → tonal / voiced speech.  AI voices tend to have higher
        flatness in the upper frequency bands because they lack natural
        breathiness and room acoustics.

        :return: {'mean_flatness': float, 'std_flatness': float}
        """
        try:
            import librosa  # noqa: PLC0415

            flatness = librosa.feature.spectral_flatness(y=waveform)  # shape: (1, T)
            mean_flat = float(np.mean(flatness))
            std_flat = float(np.std(flatness))
            logger.debug(
                "AudioAnalyzer — spectral flatness: mean=%.6f, std=%.6f",
                mean_flat,
                std_flat,
            )
            return {"mean_flatness": mean_flat, "std_flatness": std_flat}
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — spectral flatness computation failed: %s", exc)
            return {"mean_flatness": 0.0, "std_flatness": 0.0}

    @staticmethod
    def _compute_zcr_features(waveform: np.ndarray) -> dict[str, float]:
        """
        Compute zero-crossing rate (ZCR) statistics.

        ZCR captures the noisiness and fricative content of speech.
        AI voices exhibit less natural variation in ZCR across frames than
        real human voices.

        :return: {'zcr_mean': float, 'zcr_variance': float}
        """
        try:
            import librosa  # noqa: PLC0415

            zcr = librosa.feature.zero_crossing_rate(waveform)  # shape: (1, T)
            zcr_flat = zcr.flatten()
            return {
                "zcr_mean": float(np.mean(zcr_flat)),
                "zcr_variance": float(np.var(zcr_flat)),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("AudioAnalyzer — ZCR computation failed: %s", exc)
            return {"zcr_mean": 0.0, "zcr_variance": 0.0}

    # ── Verdict fusion ────────────────────────────────────────────────────────

    def _compute_verdict(
        self,
        mfcc_result: dict[str, float],
        flatness_result: dict[str, float],
        zcr_result: dict[str, float],
    ) -> tuple[float, bool, float]:
        """
        Fuse MFCC, spectral flatness, and ZCR signals into a single artifact score.

        Scoring logic:
        - MFCC variance below the threshold → synthetic signature (higher score).
        - Spectral flatness above the threshold → smooth / lacking breathiness.
        - Both features are weighted and combined into [0, 1] artifact score.
        - ZCR variance is used as a secondary corroborating signal (logged only).

        :return: (artifact_score [0,1], is_fake bool, confidence [0,99] %)
        """
        mfcc_var = mfcc_result["mean_variance"]
        flatness = flatness_result["mean_flatness"]

        # MFCC component: how far below the threshold are we?  (normalised 0→1)
        # Low variance = more synthetic → higher score component.
        if mfcc_var < self.mfcc_var_threshold:
            mfcc_component = 1.0 - (mfcc_var / self.mfcc_var_threshold)
        else:
            mfcc_component = 0.0

        # Spectral flatness component: high flatness → synthetic smoothing.
        if flatness > self.spectral_flatness_threshold:
            flatness_component = min(
                (flatness - self.spectral_flatness_threshold)
                / (1.0 - self.spectral_flatness_threshold + 1e-8),
                1.0,
            )
        else:
            flatness_component = 0.0

        # Weighted ensemble artifact score (0 = definitely real, 1 = definitely fake)
        artifact_score = (
            MFCC_WEIGHT * mfcc_component + FLATNESS_WEIGHT * flatness_component
        )

        # Decision: artifact score ≥ 0.5 → FAKE
        is_fake = artifact_score >= 0.5

        # Confidence: how far from the decision boundary scaled to [10, 99]
        _BASE = 10.0
        distance = abs(artifact_score - 0.5) * 2.0  # 0 at boundary, 1 at extremes
        confidence = _BASE + distance * (99.0 - _BASE)

        logger.debug(
            "AudioAnalyzer — mfcc_comp=%.4f, flatness_comp=%.4f, "
            "artifact_score=%.4f, zcr_var=%.8f",
            mfcc_component,
            flatness_component,
            artifact_score,
            zcr_result.get("zcr_variance", 0.0),
        )

        return artifact_score, is_fake, confidence

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup_temp(tmp_dir: str) -> None:
        """Remove the temporary directory and all its contents."""
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AudioAnalyzer — failed to remove temp dir %s: %s", tmp_dir, exc)

    @staticmethod
    def _inconclusive_result(reason: str) -> dict[str, Any]:
        """Return a standardised INCONCLUSIVE result for the pipeline ensemble."""
        return {
            "module": "Audio Forensics Analyzer",
            "is_fake": False,
            "artifact_score": 0.0,
            "confidence": 0.0,
            "mfcc_mean_variance": 0.0,
            "mfcc_variance_threshold": MFCC_VARIANCE_FAKE_THRESHOLD,
            "spectral_flatness_mean": 0.0,
            "spectral_flatness_threshold": SPECTRAL_FLATNESS_FAKE_THRESHOLD,
            "zcr_mean": 0.0,
            "zcr_variance": 0.0,
            "audio_duration_seconds": 0.0,
            "sample_rate": 0,
            "verdict": "INCONCLUSIVE",
            "reason": reason,
        }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_analyzer.py <path_to_video.mp4>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    analyzer = AudioAnalyzer()
    report = analyzer.analyze(sys.argv[1])
    import json

    print(json.dumps(report, indent=2))
