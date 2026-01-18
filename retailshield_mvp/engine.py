"""
Processing Engine - Core pipeline for retail theft detection
Coordinates all components and manages the processing flow
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque


@dataclass
class Detection:
    """Detection data class"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str = "person"


@dataclass
class Track:
    """Track data class"""
    track_id: int
    bbox: List[float]
    confidence: float
    detections: List[Detection]
    age: int = 0


@dataclass
class Pose:
    """Pose data class"""
    keypoints: List[List[float]]  # [[x, y, conf], ...]
    bbox: List[float]
    track_id: Optional[int] = None


@dataclass
class Behavior:
    """Behavior data class"""
    type: str  # "concealment", "sweep", "no_scan"
    confidence: float
    track_id: int
    description: str


@dataclass
class Alert:
    """Alert data class"""
    alert_id: str
    type: str
    confidence: float
    track_id: int
    frame_id: int
    description: str


class ProcessingEngine:
    """Main processing engine that coordinates all components"""

    def __init__(self, detector, tracker, pose_estimator, gesture_analyzer, alert_manager):
        self.logger = logging.getLogger(__name__)

        # Components
        self.detector = detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.gesture_analyzer = gesture_analyzer
        self.alert_manager = alert_manager

        # State
        self.frame_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.running = False
        self.last_time = time.time()

        self.logger.info("Processing engine initialized")

    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Process a single frame through the entire pipeline"""
        results = {
            'frame_id': frame_id,
            'detections': [],
            'tracks': [],
            'poses': [],
            'behaviors': [],
            'alerts': []
        }

        try:
            # Step 1: Detect people
            detections = self.detector.detect(frame)
            results['detections'] = detections

            # Step 2: Track detected people
            tracks = self.tracker.update(detections)
            results['tracks'] = tracks

            # Step 3: Estimate pose for each track
            poses = []
            for track in tracks:
                if len(track.detections) > 0:
                    # Use the most recent detection for pose estimation
                    latest_detection = track.detections[-1]
                    x1, y1, x2, y2 = latest_detection.bbox

                    # Extract ROI for pose estimation
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size > 0:
                        pose = self.pose_estimator.estimate(roi)
                        if pose:
                            # Adjust keypoints to original frame coordinates
                            adjusted_pose = Pose(
                                keypoints=[
                                    [x + x1, y + y1, conf]
                                    for x, y, conf in pose.keypoints
                                ],
                                bbox=track.bbox,
                                track_id=track.track_id
                            )
                            poses.append(adjusted_pose)

            results['poses'] = poses

            # Step 4: Analyze gestures and behavior
            behaviors = []
            for pose in poses:
                # Update track history
                self.track_history[pose.track_id].append(pose)

                # Analyze gestures if we have enough history
                if len(self.track_history[pose.track_id]) >= 10:
                    behavior = self.gesture_analyzer.analyze(
                        list(self.track_history[pose.track_id]),
                        pose.track_id
                    )
                    if behavior:
                        behaviors.append(behavior)

            results['behaviors'] = behaviors

            # Step 5: Generate alerts if suspicious behavior detected
            if behaviors:
                alerts = self.alert_manager.process_behaviors(behaviors, frame_id)
                results['alerts'] = alerts

                # Trigger alerts
                for alert in alerts:
                    self.alert_manager.trigger_alert(alert, frame)

            self.frame_id += 1
            return results

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_id}: {e}")
            return results

    def annotate_frame(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Annotate frame with detections, tracks, and alerts"""
        annotated_frame = frame.copy()

        # Draw tracks
        for track in results.get('tracks', []):
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0)  # Green for tracks

            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw track ID
            cv2.putText(annotated_frame, f"ID: {track.track_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw poses
        for pose in results.get('poses', []):
            for keypoint in pose.keypoints:
                x, y, conf = keypoint
                if conf > 0.5:  # Only draw confident keypoints
                    cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Draw alerts
        for alert in results.get('alerts', []):
            # Find the track for this alert
            for track in results.get('tracks', []):
                if track.track_id == alert.track_id:
                    x1, y1, x2, y2 = track.bbox

                    # Draw red alert box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

                    # Draw alert text
                    cv2.putText(annotated_frame, f"ALERT: {alert.type} ({alert.confidence:.2f})",
                                (int(x1), int(y1) - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    break

        # Draw FPS counter
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if current_time - self.last_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.last_time = current_time
        return annotated_frame

    def shutdown(self):
        """Clean shutdown of processing engine"""
        self.logger.info("Shutting down processing engine...")
        self.track_history.clear()
        self.logger.info("Processing engine shutdown complete")