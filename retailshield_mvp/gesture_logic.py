"""
Gesture Logic - Analyzes poses to detect suspicious gestures
Core logic for theft behavior detection
"""

import numpy as np
import logging
import time  # ADD THIS IMPORT AT THE TOP LEVEL
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pose import Pose

@dataclass
class GestureResult:
    """Gesture detection result"""
    type: str  # "concealment", "sweep", "no_scan", "normal"
    confidence: float
    description: str
    track_id: int = 0  # ADD THIS FIELD

class GestureAnalyzer:
    """Analyzes gestures and suspicious behaviors"""

    # Keypoint indices (MediaPipe format)
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28
    }

    def __init__(self, concealment_threshold: float = 0.7):
        self.logger = logging.getLogger(__name__)
        self.concealment_threshold = concealment_threshold

        # State for temporal analysis
        self.track_gesture_history = {}
        self.gesture_cooldown = {}

        self.logger.info("Gesture analyzer initialized")

    def analyze(self, pose_history: List[Pose], track_id: int) -> Optional[GestureResult]:
        """Analyze pose history for suspicious gestures"""
        if not pose_history or len(pose_history) < 10:
            return None

        current_pose = pose_history[-1]

        # Check for various suspicious behaviors
        behaviors = []

        # 1. Check for concealment gesture
        concealment = self._detect_concealment(current_pose, pose_history, track_id)
        if concealment:
            behaviors.append(concealment)

        # 2. Check for sweep gesture (multiple quick hand movements)
        sweep = self._detect_sweep(pose_history, track_id)
        if sweep:
            behaviors.append(sweep)

        # 3. Check for suspicious arm movements
        suspicious_arm = self._detect_suspicious_arm_movement(current_pose, pose_history, track_id)
        if suspicious_arm:
            behaviors.append(suspicious_arm)

        # Return the behavior with highest confidence
        if behaviors:
            behaviors.sort(key=lambda x: x.confidence, reverse=True)

            # Apply cooldown to prevent spam
            if track_id in self.gesture_cooldown:
                # Check if cooldown period has passed
                if time.time() - self.gesture_cooldown[track_id] < 30:  # 30 second cooldown
                    return None

            self.gesture_cooldown[track_id] = time.time()
            return behaviors[0]

        return None

    def _detect_concealment(self, current_pose: Pose, pose_history: List[Pose], track_id: int) -> Optional[GestureResult]:
        """Detect item concealment gesture (hand near body/face)"""
        if len(current_pose.keypoints) < 17:  # Need wrist keypoints
            return None

        # Get wrist and body keypoints
        left_wrist = self._get_keypoint(current_pose, 'left_wrist')
        right_wrist = self._get_keypoint(current_pose, 'right_wrist')
        left_shoulder = self._get_keypoint(current_pose, 'left_shoulder')
        right_shoulder = self._get_keypoint(current_pose, 'right_shoulder')
        left_hip = self._get_keypoint(current_pose, 'left_hip')
        right_hip = self._get_keypoint(current_pose, 'right_hip')

        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return None

        # Calculate distances from wrists to body parts
        concealment_score = 0.0
        detected_gestures = []

        # Check if wrist is near shoulder (concealing in jacket/upper body)
        left_wrist_to_shoulder = self._distance(left_wrist[:2], left_shoulder[:2])
        right_wrist_to_shoulder = self._distance(right_wrist[:2], right_shoulder[:2])

        if left_wrist_to_shoulder < 50:  # pixels
            concealment_score += 0.4
            detected_gestures.append("left hand near shoulder")

        if right_wrist_to_shoulder < 50:
            concealment_score += 0.4
            detected_gestures.append("right hand near shoulder")

        # Check if wrist is near hip (concealing in pocket)
        if left_hip and right_hip:
            left_wrist_to_hip = self._distance(left_wrist[:2], left_hip[:2])
            right_wrist_to_hip = self._distance(right_wrist[:2], right_hip[:2])

            if left_wrist_to_hip < 40:
                concealment_score += 0.3
                detected_gestures.append("left hand near hip")

            if right_wrist_to_hip < 40:
                concealment_score += 0.3
                detected_gestures.append("right hand near hip")

        # Check for hands together (might be transferring item)
        wrist_distance = self._distance(left_wrist[:2], right_wrist[:2])
        if wrist_distance < 30:
            concealment_score += 0.3
            detected_gestures.append("hands together")

        # Check if this pose is sustained over time
        if len(pose_history) >= 10:
            # Count how many recent frames show concealment
            concealment_frames = 0
            for pose in pose_history[-10:]:
                if self._quick_concealment_check(pose):
                    concealment_frames += 1

            if concealment_frames >= 7:  # 70% of recent frames
                concealment_score += 0.2

        # Normalize score to 0-1
        concealment_score = min(1.0, concealment_score)

        if concealment_score >= self.concealment_threshold and detected_gestures:
            description = f"Possible concealment: {', '.join(detected_gestures)}"
            return GestureResult(
                type="concealment",
                confidence=concealment_score,
                description=description,
                track_id=track_id
            )

        return None

    def _quick_concealment_check(self, pose: Pose) -> bool:
        """Quick check for concealment in a single pose"""
        if len(pose.keypoints) < 17:
            return False

        left_wrist = self._get_keypoint(pose, 'left_wrist')
        right_wrist = self._get_keypoint(pose, 'right_wrist')
        left_shoulder = self._get_keypoint(pose, 'left_shoulder')
        right_shoulder = self._get_keypoint(pose, 'right_shoulder')

        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False

        # Check if either wrist is near corresponding shoulder
        left_dist = self._distance(left_wrist[:2], left_shoulder[:2])
        right_dist = self._distance(right_wrist[:2], right_shoulder[:2])

        return left_dist < 60 or right_dist < 60

    def _detect_sweep(self, pose_history: List[Pose], track_id: int) -> Optional[GestureResult]:
        """Detect sweep gesture (multiple quick hand movements)"""
        if len(pose_history) < 15:
            return None

        # Analyze hand movement speed and patterns
        recent_poses = pose_history[-15:]

        # Calculate hand movement variance
        wrist_positions = []
        for pose in recent_poses:
            left_wrist = self._get_keypoint(pose, 'left_wrist')
            right_wrist = self._get_keypoint(pose, 'right_wrist')

            if left_wrist and right_wrist:
                wrist_positions.append((left_wrist[0], left_wrist[1], right_wrist[0], right_wrist[1]))

        if len(wrist_positions) < 10:
            return None

        # Calculate movement metrics
        movements = []
        for i in range(1, len(wrist_positions)):
            prev = wrist_positions[i-1]
            curr = wrist_positions[i]

            left_movement = self._distance((prev[0], prev[1]), (curr[0], curr[1]))
            right_movement = self._distance((prev[2], prev[3]), (curr[2], curr[3]))

            avg_movement = (left_movement + right_movement) / 2
            movements.append(avg_movement)

        # Check for rapid, repetitive movements
        if len(movements) >= 5:
            avg_movement = np.mean(movements)
            movement_var = np.var(movements)

            # High average movement with low variance indicates consistent rapid movement
            if avg_movement > 20 and movement_var < 100:
                return GestureResult(
                    type="sweep",
                    confidence=min(0.8, avg_movement / 50),  # Normalize
                    description="Rapid hand movements detected (possible sweep)",
                    track_id=track_id
                )

        return None

    def _detect_suspicious_arm_movement(self, current_pose: Pose, pose_history: List[Pose], track_id: int) -> Optional[GestureResult]:
        """Detect suspicious arm movements (unusual angles/extensions)"""
        if len(current_pose.keypoints) < 17:
            return None

        # Get keypoints for arm angles
        left_shoulder = self._get_keypoint(current_pose, 'left_shoulder')
        left_elbow = self._get_keypoint(current_pose, 'left_elbow')
        left_wrist = self._get_keypoint(current_pose, 'left_wrist')

        right_shoulder = self._get_keypoint(current_pose, 'right_shoulder')
        right_elbow = self._get_keypoint(current_pose, 'right_elbow')
        right_wrist = self._get_keypoint(current_pose, 'right_wrist')

        suspicious_score = 0.0
        detected_gestures = []

        # Calculate arm angles
        if left_shoulder and left_elbow and left_wrist:
            left_angle = self._calculate_angle(left_shoulder[:2], left_elbow[:2], left_wrist[:2])

            # Check for unusual arm extension (reaching)
            if left_angle > 160:  # Nearly straight arm
                suspicious_score += 0.3
                detected_gestures.append("left arm extended")

        if right_shoulder and right_elbow and right_wrist:
            right_angle = self._calculate_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])

            if right_angle > 160:
                suspicious_score += 0.3
                detected_gestures.append("right arm extended")

        # Check for arms raised above shoulders
        if left_shoulder and left_wrist:
            if left_wrist[1] < left_shoulder[1] - 20:  # Wrist above shoulder
                suspicious_score += 0.2
                detected_gestures.append("left arm raised")

        if right_shoulder and right_wrist:
            if right_wrist[1] < right_shoulder[1] - 20:
                suspicious_score += 0.2
                detected_gestures.append("right arm raised")

        if suspicious_score > 0.5 and detected_gestures:
            return GestureResult(
                type="suspicious_arm_movement",
                confidence=min(0.7, suspicious_score),
                description=f"Suspicious arm movements: {', '.join(detected_gestures)}",
                track_id=track_id
            )

        return None

    def _get_keypoint(self, pose: Pose, keypoint_name: str) -> Optional[List[float]]:
        """Get keypoint by name"""
        if keypoint_name in self.KEYPOINT_INDICES:
            idx = self.KEYPOINT_INDICES[keypoint_name]
            if idx < len(pose.keypoints):
                return pose.keypoints[idx]
        return None

    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _calculate_angle(self, a, b, c):
        """Calculate angle at point b formed by points a-b-c"""
        import math

        # Convert to numpy arrays for easier calculation
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1, 1))

        return np.degrees(angle)