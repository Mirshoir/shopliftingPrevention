"""
Object Tracker - Tracks detected people across frames
Simple implementation of tracking algorithm
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple
from detector import Detection


@dataclass
class Track:
    """Track data class"""
    track_id: int
    bbox: List[float]
    confidence: float
    detections: List[Detection]
    age: int = 0
    hits: int = 1


class ObjectTracker:
    """Simple object tracker using Hungarian algorithm"""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.logger = logging.getLogger(__name__)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        # State
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0

        self.logger.info(f"Tracker initialized (max_age={max_age}, min_hits={min_hits})")

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections"""
        self.frame_count += 1

        # Update existing tracks
        self._update_existing_tracks()

        # Associate detections with tracks
        if len(detections) > 0:
            matched_pairs = self._associate_detections_to_tracks(detections)
            self._update_matched_tracks(matched_pairs, detections)

        # Create new tracks for unmatched detections
        self._create_new_tracks(detections)

        # Remove dead tracks
        self._remove_dead_tracks()

        return self.tracks

    def _update_existing_tracks(self):
        """Update age of existing tracks"""
        for track in self.tracks:
            track.age += 1

    def _associate_detections_to_tracks(self, detections: List[Detection]) -> List[Tuple[int, int]]:
        """Match detections to tracks using IoU"""
        if len(self.tracks) == 0 or len(detections) == 0:
            return []

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)

        # Find matches with IoU above threshold
        matched_pairs = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    matched_pairs.append((i, j))

        # Simple greedy matching (for MVP)
        # Sort by IoU descending
        matched_pairs.sort(key=lambda x: iou_matrix[x[0], x[1]], reverse=True)

        # Keep only one match per detection/track
        final_matches = []
        used_tracks = set()
        used_detections = set()

        for i, j in matched_pairs:
            if i not in used_tracks and j not in used_detections:
                final_matches.append((i, j))
                used_tracks.add(i)
                used_detections.add(j)

        return final_matches

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _update_matched_tracks(self, matched_pairs: List[Tuple[int, int]], detections: List[Detection]):
        """Update tracks that matched with detections"""
        for track_idx, detection_idx in matched_pairs:
            track = self.tracks[track_idx]
            detection = detections[detection_idx]

            # Update track with new detection (simple averaging)
            alpha = 0.3  # Learning rate
            track.bbox = [
                (1 - alpha) * track.bbox[0] + alpha * detection.bbox[0],
                (1 - alpha) * track.bbox[1] + alpha * detection.bbox[1],
                (1 - alpha) * track.bbox[2] + alpha * detection.bbox[2],
                (1 - alpha) * track.bbox[3] + alpha * detection.bbox[3],
            ]
            track.confidence = (track.confidence * track.hits + detection.confidence) / (track.hits + 1)
            track.detections.append(detection)
            track.hits += 1
            track.age = 0  # Reset age

    def _create_new_tracks(self, detections: List[Detection]):
        """Create new tracks for unmatched detections"""
        # Find unmatched detections
        matched_detection_indices = set()
        for track in self.tracks:
            if hasattr(track, 'last_detection_idx'):
                matched_detection_indices.add(track.last_detection_idx)

        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                # Create new track
                new_track = Track(
                    track_id=self.next_id,
                    bbox=detection.bbox.copy(),
                    confidence=detection.confidence,
                    detections=[detection]
                )
                self.tracks.append(new_track)
                self.next_id += 1

    def _remove_dead_tracks(self):
        """Remove tracks that are too old or never confirmed"""
        alive_tracks = []
        for track in self.tracks:
            # Remove if too old
            if track.age > self.max_age:
                self.logger.debug(f"Removing track {track.track_id} (age: {track.age})")
                continue

            # Remove if never confirmed (min_hits not reached)
            if track.hits < self.min_hits and track.age > 5:
                self.logger.debug(f"Removing unconfirmed track {track.track_id}")
                continue

            alive_tracks.append(track)

        self.tracks = alive_tracks