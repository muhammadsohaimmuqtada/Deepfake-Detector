import cv2
import numpy as np
import logging

# Setup logging for enterprise-level tracking
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Fraction of the 1-D spectrum used for the low-frequency and high-frequency bands.
# The low band covers the first SPECTRAL_BAND_RATIO of the radial profile;
# the high band covers the last SPECTRAL_BAND_RATIO.
SPECTRAL_BAND_RATIO = 0.30

# Minimum face crop dimension (pixels). Crops smaller than this are extremely
# pixelated and produce artificially inflated variance — skip them.
MIN_FACE_SIZE = 32


class FrequencyAnalyzer:
    def __init__(self, high_freq_threshold=0.60):
        """
        Initializes the FrequencyAnalyzer.

        :param high_freq_threshold: Spectral Energy Ratio above which a face crop is
            flagged as synthetic.  The ratio is high_band_mean / low_band_mean
            after shifting the log-magnitude PSD to be strictly positive.
            Natural images have a ratio well below 0.60; GAN-generated images
            (with upsampling grid artefacts) exceed this value.
        """
        self.threshold = high_freq_threshold

    def azimuthal_average(self, image, center=None):
        """
        Calculates the 1D power spectrum of a 2D Fourier transform.
        This collapses the 2D frequency map into a 1D graph to easily spot GAN upsampling artifacts.
        """
        y, x = np.indices((image.shape))
        if not center:
            center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        
        r = np.hypot(x - center[0], y - center[1])
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]

        # Get the integer part of the radii (bins)
        r_int = r_sorted.astype(int)
        
        # Find all pixels that fall within each radial bin
        deltar = r_int[1:] - r_int[:-1]
        rind = np.where(deltar)[0]
        nr = rind[1:] - rind[:-1]
        
        # Cumulative sum to figure out the mean for each radius
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        radial_prof = tbin / nr

        return radial_prof

    def detect_artifacts(self, face_image):
        """
        Analyzes a face image for synthetic generation artifacts using FFT.

        The key metric is the **Spectral Energy Ratio**: the mean energy in the
        high-frequency band divided by the mean energy in the low-frequency band.
        Natural images follow a steep 1/f^n power decay, so this ratio is low.
        GAN-generated images suffer from upsampling grid artefacts that
        artificially elevate high-frequency energy, raising the ratio.

        :param face_image: NumPy array of the cropped face (BGR or Grayscale).
        :return: Dict containing the result and confidence score.
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        h, w = gray.shape

        # Fallback: extremely small crops are heavily pixelated and produce
        # artificially elevated variance that would skew the score.
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            logging.warning(
                "Face crop too small (%dx%d px) — skipping FFT analysis.", w, h
            )
            return {
                "is_fake": False,
                "artifact_score": 0.0,
                "confidence": 0.0,
                "module": "Frequency/FFT Analyzer",
                "verdict": "INCONCLUSIVE",
                "reason": "face_too_small",
            }

        # 1. Apply Fast Fourier Transform (FFT)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 2. Log-magnitude spectrum (dB scale)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # 3. 1D Azimuthal Average → radial power profile
        # Real images decay smoothly from centre outward; GANs have abnormal
        # bumps in the high-frequency tail.
        psd_1d = self.azimuthal_average(magnitude_spectrum)

        # 4. Spectral Energy Ratio
        # Shift the profile to be strictly positive before computing the ratio
        # (log-magnitudes can be negative for near-zero amplitudes).
        psd_positive = psd_1d - np.min(psd_1d) + 1.0  # +1.0: shift to strictly positive (log-magnitudes can go negative)
        n = len(psd_positive)
        n_band = max(1, int(n * SPECTRAL_BAND_RATIO))

        low_band_mean = np.mean(psd_positive[:n_band])    # mean energy in the low-frequency band
        high_band_mean = np.mean(psd_positive[-n_band:])  # mean energy in the high-frequency band

        # Ratio: natural images → low value; GANs → elevated value
        noise_score = high_band_mean / (low_band_mean + 1e-8)

        logging.info(
            "Spectral Energy Ratio (FFT Score): %.4f  (low=%.2f, high=%.2f)",
            noise_score, low_band_mean, high_band_mean,
        )

        # 5. Make a determination based on the threshold
        is_fake = bool(noise_score > self.threshold)
        if is_fake:
            confidence = min((noise_score / self.threshold) * 50, 99.0)
        else:
            confidence = 99.0 - min((noise_score / self.threshold) * 50, 99.0)

        return {
            "is_fake": is_fake,
            "artifact_score": float(noise_score),
            "confidence": float(confidence),
            "module": "Frequency/FFT Analyzer",
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    # Create a dummy "noisy" image to simulate a deepfake artifact test
    dummy_fake_face = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    detector = FrequencyAnalyzer()
    result = detector.detect_artifacts(dummy_fake_face)
    print(f"Test Result: {result}")
