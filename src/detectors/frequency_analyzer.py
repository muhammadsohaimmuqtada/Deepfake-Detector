import cv2
import numpy as np
import logging

# Setup logging for enterprise-level tracking
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

HIGH_FREQ_TAIL_RATIO = 0.2  # Fraction of the spectrum considered high-frequency


class FrequencyAnalyzer:
    def __init__(self, high_freq_threshold=0.15):
        """
        Initializes the FrequencyAnalyzer.
        :param high_freq_threshold: The threshold above which high-frequency noise is flagged as synthetic.
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
        :param face_image: NumPy array of the cropped face (BGR or Grayscale).
        :return: Dict containing the result and confidence score.
        """
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # 1. Apply Fast Fourier Transform (FFT)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # 2. Calculate the magnitude spectrum (Power spectrum)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        # 3. Calculate 1D Azimuthal Average
        # Real images decay smoothly. GANs/Deepfakes have abnormal bumps in the high-frequency tail.
        psd_1d = self.azimuthal_average(magnitude_spectrum)
        
        # 4. Analyze High Frequencies (The "Deepfake Signature" zone)
        # We look at the last HIGH_FREQ_TAIL_RATIO of the spectrum (the highest frequencies)
        high_freq_tail = psd_1d[-int(len(psd_1d) * HIGH_FREQ_TAIL_RATIO):]
        mean_tail = np.mean(high_freq_tail)
        noise_score = np.var(high_freq_tail) / (mean_tail + 1e-8)

        logging.info(f"Calculated High-Frequency Artifact Score: {noise_score:.4f}")

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
            "module": "Frequency/FFT Analyzer"
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    # Create a dummy "noisy" image to simulate a deepfake artifact test
    dummy_fake_face = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    detector = FrequencyAnalyzer()
    result = detector.detect_artifacts(dummy_fake_face)
    print(f"Test Result: {result}")
