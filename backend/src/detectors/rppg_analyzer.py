"""
RPPGAnalyzer — Layer 2 Forensic Module: Remote Photoplethysmography (rPPG) Heartbeat Analyzer.

Core Principle (The AI Flaw):
    Deepfake generators synthesize pixel values frame-by-frame without modeling
    the microscopic, periodic color fluctuations caused by blood flowing through
    facial capillaries.  A real human face exhibits a subtle, rhythmic green-channel
    intensity oscillation at a frequency corresponding to heart rate (42–180 BPM /
    0.7–3.0 Hz).  Fakes have chaotic, non-periodic noise instead.

Algorithm:
    1. Extract the mean Green-channel value from each face frame → 1-D temporal signal.
    2. Apply a Butterworth bandpass filter (0.7–3.0 Hz) to isolate heart-rate band.
    3. Compute FFT of the filtered signal to find the dominant frequency.
    4. Evaluate peak-to-noise ratio (SNR) to decide real vs. synthetic.

Usage:
    from src.detectors.rppg_analyzer import RPPGAnalyzer
    analyzer = RPPGAnalyzer(fps=30)
    result = analyzer.analyze(face_frames)   # face_frames: list of BGR NumPy arrays
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

# ── Signal quality constants ──────────────────────────────────────────────────
HEART_RATE_LOW_HZ = 0.75   # 45 BPM lower bound
HEART_RATE_HIGH_HZ = 3.0   # 180 BPM upper bound
FILTER_ORDER = 5            # Butterworth filter order
MIN_FRAMES = 15             # Minimum frames required for a meaningful analysis
SNR_FAKE_THRESHOLD = 2.0    # Peak-SNR below this → likely synthetic


class RPPGAnalyzer:
    """
    Remote Photoplethysmography (rPPG) analyzer.

    Detects the presence (or absence) of a plausible human heartbeat signal
    in a temporal sequence of cropped face images.
    """

    def __init__(self, fps: float = 30.0) -> None:
        """
        :param fps: Frames-per-second of the source video (after frame-skip).
                    Used to construct the correct bandpass filter cutoffs.
        """
        self.fps = fps
        self._sos = self._build_bandpass_filter()
        logger.info(
            "RPPGAnalyzer ready — fps=%.1f, passband=[%.1f–%.1f Hz]",
            fps,
            HEART_RATE_LOW_HZ,
            HEART_RATE_HIGH_HZ,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, face_frames: list[np.ndarray]) -> dict[str, Any]:
        """
        Analyze a temporal sequence of face crops for a heartbeat signal.

        :param face_frames: Ordered list of BGR (or RGB) NumPy arrays, each
                            representing one cropped face from a consecutive frame.
        :return: Forensic result dictionary compatible with pipeline ensemble.
        """
        n = len(face_frames)

        if n < MIN_FRAMES:
            logger.warning(
                "rPPG: insufficient frames (%d < %d required) — returning INCONCLUSIVE.",
                n,
                MIN_FRAMES,
            )
            return self._inconclusive_result(n, reason="insufficient_frames")

        # Step 1 — Extract Green-channel mean per frame → raw rPPG signal
        raw_signal = self._extract_green_signal(face_frames)

        # Step 2 — Detrend (remove slow illumination drift)
        detrended = raw_signal - np.mean(raw_signal)

        # Step 3 — Bandpass filter (isolate 0.7–3.0 Hz)
        try:
            filtered = sosfilt(self._sos, detrended)
        except Exception as exc:  # noqa: BLE001
            logger.error("rPPG bandpass filter failed: %s", exc)
            return self._inconclusive_result(n, reason="filter_error")

        # Step 4 — FFT → frequency-domain analysis
        snr, dominant_hz, dominant_bpm = self._frequency_analysis(filtered)

        # Step 5 — Verdict
        is_fake = snr < SNR_FAKE_THRESHOLD
        confidence = self._compute_confidence(snr, is_fake)

        logger.info(
            "rPPG result — dominant=%.2f Hz (%.0f BPM), SNR=%.3f, is_fake=%s, confidence=%.1f%%",
            dominant_hz,
            dominant_bpm,
            snr,
            is_fake,
            confidence,
        )

        return {
            "module": "rPPG Heartbeat Analyzer",
            "is_fake": is_fake,
            "confidence": round(confidence, 2),
            "rppg_snr": round(float(snr), 4),
            "dominant_frequency_hz": round(float(dominant_hz), 3),
            "dominant_bpm": round(float(dominant_bpm), 1),
            "frames_analyzed": n,
            "snr_threshold": SNR_FAKE_THRESHOLD,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_bandpass_filter(self):
        """
        Construct a zero-phase Butterworth bandpass filter using second-order
        sections (SOS) for numerical stability.
        """
        nyquist = self.fps / 2.0
        low = HEART_RATE_LOW_HZ / nyquist
        high = HEART_RATE_HIGH_HZ / nyquist
        # Clamp to valid range (0, 1) exclusive — safety guard for unusual fps values
        low = max(1e-4, min(low, 0.99))
        high = max(low + 1e-4, min(high, 0.99))
        sos = butter(FILTER_ORDER, [low, high], btype="bandpass", output="sos")
        return sos

    @staticmethod
    def _extract_green_signal(face_frames: list[np.ndarray]) -> np.ndarray:
        """
        Extract the spatial mean of the Green channel for each frame.

        OpenCV stores images as BGR; index 1 is Green in both BGR and RGB.
        """
        signal = np.empty(len(face_frames), dtype=np.float64)
        for i, frame in enumerate(face_frames):
            if frame.ndim == 3 and frame.shape[2] >= 3:
                green_channel = frame[:, :, 1].astype(np.float64)
            elif frame.ndim == 2:
                # Grayscale fallback — use the whole channel as a proxy
                green_channel = frame.astype(np.float64)
            else:
                green_channel = np.zeros((1,), dtype=np.float64)
            signal[i] = green_channel.mean()
        return signal

    def _frequency_analysis(self, signal: np.ndarray) -> tuple[float, float, float]:
        """
        Run FFT on the filtered signal and calculate the peak SNR.

        :return: (snr, dominant_hz, dominant_bpm)
        """
        n = len(signal)
        # Windowing reduces spectral leakage
        windowed = signal * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)

        # Focus only on the heart-rate band
        band_mask = (freqs >= HEART_RATE_LOW_HZ) & (freqs <= HEART_RATE_HIGH_HZ)
        if not np.any(band_mask):
            return 0.0, 0.0, 0.0

        band_spectrum = spectrum[band_mask]
        band_freqs = freqs[band_mask]

        peak_idx = np.argmax(band_spectrum)
        peak_power = band_spectrum[peak_idx]
        dominant_hz = float(band_freqs[peak_idx])

        # SNR = peak power / mean of remaining band power
        other_power = np.delete(band_spectrum, peak_idx)
        noise_floor = np.mean(other_power) if other_power.size > 0 else 1e-8
        snr = float(peak_power / (noise_floor + 1e-8))

        dominant_bpm = dominant_hz * 60.0
        return snr, dominant_hz, dominant_bpm

    @staticmethod
    def _compute_confidence(snr: float, is_fake: bool) -> float:
        """
        Map SNR to a [0, 99] confidence percentage.

        When is_fake=True  → higher SNR gap below threshold = higher fake confidence.
        When is_fake=False → higher SNR above threshold = higher real confidence.

        A small baseline offset ensures non-zero confidence at the decision boundary
        (SNR exactly at SNR_FAKE_THRESHOLD).
        """
        _BASE = 10.0  # minimum confidence at the decision boundary
        if is_fake:
            # How far below the threshold are we?  Normalise to [_BASE, 99].
            deficit = max(SNR_FAKE_THRESHOLD - snr, 0.0)
            confidence = _BASE + min((deficit / SNR_FAKE_THRESHOLD) * (99.0 - _BASE), 99.0 - _BASE)
        else:
            surplus = max(snr - SNR_FAKE_THRESHOLD, 0.0)
            confidence = _BASE + min((surplus / SNR_FAKE_THRESHOLD) * (99.0 - _BASE), 99.0 - _BASE)
        return float(confidence)

    @staticmethod
    def _inconclusive_result(n: int, reason: str) -> dict[str, Any]:
        return {
            "module": "rPPG Heartbeat Analyzer",
            "is_fake": False,
            "confidence": 0.0,
            "rppg_snr": 0.0,
            "dominant_frequency_hz": 0.0,
            "dominant_bpm": 0.0,
            "frames_analyzed": n,
            "snr_threshold": SNR_FAKE_THRESHOLD,
            "verdict": "INCONCLUSIVE",
            "reason": reason,
        }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2  # noqa: PLC0415

    fps = 30
    analyzer = RPPGAnalyzer(fps=fps)
    n_frames = 60

    # Simulate a REAL face: embed a 1 Hz sinusoidal heartbeat in the green channel
    real_frames = []
    for i in range(n_frames):
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        heartbeat_val = int(10 * np.sin(2 * np.pi * 1.0 * i / fps))  # 60 BPM
        frame[:, :, 1] = np.clip(128 + heartbeat_val, 0, 255)
        real_frames.append(frame)

    # Simulate a FAKE face: pure random noise (no heartbeat)
    rng = np.random.default_rng(seed=42)
    fake_frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(n_frames)]

    real_result = analyzer.analyze(real_frames)
    fake_result = analyzer.analyze(fake_frames)

    print("=== REAL face signal ===")
    print(real_result)
    print("\n=== FAKE face signal ===")
    print(fake_result)
