import cv2
import mediapipe as mp
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

FACE_PADDING_RATIO = 0.10  # 10% padding around the bounding box to capture jawline blending artifacts


class VideoExtractor:
    """
    Extracts cropped face images from a video file using MediaPipe face detection.

    Uses a frame_skip parameter for performance optimization, processing only every
    Nth frame. Adds a 10% bounding box padding to ensure blending artifacts along
    the jawline are captured for downstream frequency analysis.
    """

    def __init__(self, frame_skip: int = 5):
        """
        Initializes the VideoExtractor.

        :param frame_skip: Process every Nth frame to reduce compute overhead.
                           Deepfake artifacts persist across frames, so full-fps
                           analysis is unnecessary.
        """
        self.frame_skip = frame_skip

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,           # 1 = full-range model (up to ~5 m)
            min_detection_confidence=0.75
        )
        logging.info(
            "VideoExtractor initialized — frame_skip=%d, min_confidence=0.75",
            self.frame_skip,
        )

    def extract_faces_from_video(self, video_path: str):
        """
        Reads a video file, detects faces in sampled frames, and yields
        padded, cropped face images as NumPy arrays (BGR).

        :param video_path: Absolute or relative path to the video file.
        :yields: NumPy ndarray — cropped face region (BGR, uint8).
        :raises FileNotFoundError: If the video file does not exist.
        """
        if not os.path.exists(video_path):
            logging.error("Video file not found: %s", video_path)
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Failed to open video: %s", video_path)
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_count = 0
        faces_extracted = 0
        logging.info("Starting face extraction: %s", video_path)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)

                if not results.detections:
                    continue

                ih, iw, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    x = int(bbox.xmin * iw)
                    y = int(bbox.ymin * ih)
                    w = int(bbox.width * iw)
                    h = int(bbox.height * ih)

                    # Add 10% padding to capture jawline/blending-edge artifacts
                    pad_x = int(w * FACE_PADDING_RATIO)
                    pad_y = int(h * FACE_PADDING_RATIO)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(iw, x + w + pad_x)
                    y2 = min(ih, y + h + pad_y)

                    cropped_face = frame[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        faces_extracted += 1
                        yield cropped_face
        finally:
            cap.release()
            logging.info(
                "Extraction complete — frames processed: %d, faces extracted: %d",
                frame_count,
                faces_extracted,
            )


# --- Quick Test Block ---
if __name__ == "__main__":
    import sys

    test_video = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    if not os.path.exists(test_video):
        print(f"Usage: python src/utils/video_extractor.py <path_to_video>")
        print("Please provide a valid video path to test the extractor.")
        raise SystemExit(1)

    extractor = VideoExtractor(frame_skip=10)
    for i, face in enumerate(extractor.extract_faces_from_video(test_video)):
        print(f"Extracted face {i}: shape={face.shape}")
