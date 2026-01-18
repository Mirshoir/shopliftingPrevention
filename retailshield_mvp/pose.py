"""
Pose Estimator - Estimates human body keypoints
Simple implementation using MediaPipe or OpenPose-like logic
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Pose:
    """Pose data class"""
    keypoints: List[List[float]]  # [[x, y, conf], ...]
    bbox: List[float]


class PoseEstimator:
    """Pose estimation using MediaPipe or simple heuristic methods"""

    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.use_mediapipe = False
        self.use_ultralytics = False

        self.initialize()

    def initialize(self):
        """Initialize pose estimation model"""
        self.logger.info("Initializing pose estimator...")

        # Try MediaPipe first (fastest, no GPU needed)
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            self.logger.info("Using MediaPipe for pose estimation")
            return
        except ImportError:
            self.logger.warning("MediaPipe not available")

        # Try Ultralytics YOLO-Pose
        if self.model_path and Path(self.model_path).exists():
            try:
                from ultralytics import YOLO
                self.yolo_pose = YOLO(self.model_path)
                self.use_ultralytics = True
                self.logger.info(f"Using YOLO-Pose from {self.model_path}")
                return
            except ImportError:
                self.logger.warning("Ultralytics YOLO not available")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO-Pose: {e}")

        # Fallback to simple heuristic method
        self.logger.info("Using heuristic pose estimation (fallback)")
        self.use_heuristic = True

    def estimate(self, frame: np.ndarray) -> Optional[Pose]:
        """Estimate pose in the given frame"""
        if frame.size == 0:
            return None

        try:
            if self.use_mediapipe:
                return self._estimate_mediapipe(frame)
            elif self.use_ultralytics:
                return self._estimate_yolo_pose(frame)
            else:
                return self._estimate_heuristic(frame)
        except Exception as e:
            self.logger.error(f"Pose estimation error: {e}")
            return None

    def _estimate_mediapipe(self, frame: np.ndarray) -> Optional[Pose]:
        """Estimate pose using MediaPipe"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Process the image
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = []
        height, width = frame.shape[:2]

        for landmark in results.pose_landmarks.landmark:
            # MediaPipe gives normalized coordinates
            x = landmark.x * width
            y = landmark.y * height
            conf = landmark.visibility  # MediaPipe uses visibility instead of confidence

            keypoints.append([x, y, conf])

        # Calculate bounding box from keypoints
        if keypoints:
            xs = [kp[0] for kp in keypoints if kp[2] > 0.1]
            ys = [kp[1] for kp in keypoints if kp[2] > 0.1]

            if xs and ys:
                bbox = [
                    min(xs) - 10,  # Add padding
                    min(ys) - 10,
                    max(xs) + 10,
                    max(ys) + 10
                ]

                return Pose(keypoints=keypoints, bbox=bbox)

        return None

    def _estimate_yolo_pose(self, frame: np.ndarray) -> Optional[Pose]:
        """Estimate pose using YOLO-Pose"""
        results = self.yolo_pose(frame, verbose=False)

        if len(results) == 0:
            return None

        # Get the first result
        result = results[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None

        # Extract keypoints from first person
        keypoints_data = result.keypoints.data[0].cpu().numpy()

        keypoints = []
        for i in range(keypoints_data.shape[0]):
            x, y, conf = keypoints_data[i]
            keypoints.append([float(x), float(y), float(conf)])

        # Get bounding box if available
        if result.boxes is not None and len(result.boxes) > 0:
            bbox = result.boxes.xyxy[0].cpu().numpy().tolist()
        else:
            # Estimate bbox from keypoints
            xs = [kp[0] for kp in keypoints if kp[2] > 0.1]
            ys = [kp[1] for kp in keypoints if kp[2] > 0.1]

            if xs and ys:
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                bbox = [0, 0, frame.shape[1], frame.shape[0]]

        return Pose(keypoints=keypoints, bbox=bbox)

    def _estimate_heuristic(self, frame: np.ndarray) -> Optional[Pose]:
        """Heuristic pose estimation (simple fallback)"""
        height, width = frame.shape[:2]

        # Generate dummy keypoints (for testing/demo)
        # This is just for MVP - in production, use proper pose estimation
        center_x = width // 2
        center_y = height // 2

        # Create simple body keypoints
        keypoints = []

        # Head
        keypoints.append([center_x, center_y - height // 4, 0.9])

        # Shoulders
        keypoints.append([center_x - width // 6, center_y - height // 8, 0.8])
        keypoints.append([center_x + width // 6, center_y - height // 8, 0.8])

        # Elbows
        keypoints.append([center_x - width // 3, center_y, 0.7])
        keypoints.append([center_x + width // 3, center_y, 0.7])

        # Wrists
        keypoints.append([center_x - width // 2, center_y + height // 8, 0.6])
        keypoints.append([center_x + width // 2, center_y + height // 8, 0.6])

        # Hips
        keypoints.append([center_x - width // 8, center_y + height // 4, 0.8])
        keypoints.append([center_x + width // 8, center_y + height // 4, 0.8])

        # Knees
        keypoints.append([center_x - width // 8, center_y + height // 2, 0.7])
        keypoints.append([center_x + width // 8, center_y + height // 2, 0.7])

        # Ankles
        keypoints.append([center_x - width // 8, center_y + height * 3 // 4, 0.6])
        keypoints.append([center_x + width // 8, center_y + height * 3 // 4, 0.6])

        # Bounding box
        bbox = [
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2
        ]

        return Pose(keypoints=keypoints, bbox=bbox)

    def draw_pose(self, frame: np.ndarray, pose: Pose) -> np.ndarray:
        """Draw pose keypoints and connections on frame"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]

        # Define connections (body parts)
        connections = [
            # Head to shoulders
            (0, 1), (0, 2),
            # Shoulders to elbows
            (1, 3), (2, 4),
            # Elbows to wrists
            (3, 5), (4, 6),
            # Shoulders to hips
            (1, 7), (2, 8),
            # Hips to knees
            (7, 9), (8, 10),
            # Knees to ankles
            (9, 11), (10, 12),
            # Hips connection
            (7, 8)
        ]

        # Draw connections
        for i, j in connections:
            if i < len(pose.keypoints) and j < len(pose.keypoints):
                kp1 = pose.keypoints[i]
                kp2 = pose.keypoints[j]

                if kp1[2] > 0.1 and kp2[2] > 0.1:  # Confidence threshold
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for kp in pose.keypoints:
            if kp[2] > 0.1:
                center = (int(kp[0]), int(kp[1]))
                cv2.circle(frame_copy, center, 4, (0, 0, 255), -1)

        return frame_copy