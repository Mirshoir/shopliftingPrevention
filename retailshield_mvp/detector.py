"""
Person Detector - YOLO-based person detection
"""

import cv2
import numpy as np
import torch
import logging
from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    """Detection data class"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str = "person"


class PersonDetector:
    """YOLO-based person detector"""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.55):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None

        self.initialize()

    def initialize(self):
        """Initialize the detector model"""
        self.logger.info(f"Initializing person detector with model: {self.model_path}")

        try:
            # Check for GPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info("Using GPU for detection")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU for detection")

            # Try to load Ultralytics YOLO
            try:
                from ultralytics import YOLO

                model_file = Path(self.model_path)
                if not model_file.exists():
                    self.logger.info(f"Downloading YOLO model: {self.model_path}")
                    self.model = YOLO(self.model_path)  # This will download if not exists
                else:
                    self.model = YOLO(self.model_path)

                self.logger.info("Loaded YOLO model successfully")
                self.use_yolo = True

            except ImportError:
                self.logger.warning("Ultralytics not installed, using OpenCV DNN as fallback")
                self._setup_opencv_dnn()

        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {e}")
            raise

    def _setup_opencv_dnn(self):
        """Setup OpenCV DNN as fallback"""
        try:
            # Use a lightweight model for faster performance
            config_path = "yolov3-tiny.cfg"
            weights_path = "yolov3-tiny.weights"
            names_path = "coco.names"

            # Check if files exist, download if not
            if not Path(config_path).exists():
                self.logger.info("Downloading YOLO tiny configuration...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                    config_path
                )

            if not Path(weights_path).exists():
                self.logger.info("Downloading YOLO tiny weights...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://pjreddie.com/media/files/yolov3-tiny.weights",
                    weights_path
                )

            if not Path(names_path).exists():
                self.logger.info("Downloading COCO names...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
                    names_path
                )

            # Load network
            self.net = cv2.dnn.readNet(weights_path, config_path)

            # Try to use CUDA, fallback to CPU
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.logger.info("Using CUDA for OpenCV DNN")
            except:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("Using CPU for OpenCV DNN")

            # Load class names
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

            # Get output layer names
            self.layer_names = self.net.getLayerNames()
            try:
                # OpenCV 4.x
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                # OpenCV 3.x
                self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

            self.use_opencv_dnn = True
            self.logger.info("OpenCV DNN setup complete")

        except Exception as e:
            self.logger.error(f"Failed to setup OpenCV DNN: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect people in a frame"""
        detections = []

        try:
            if hasattr(self, 'use_opencv_dnn') and self.use_opencv_dnn:
                detections = self._detect_opencv_dnn(frame)
            else:
                detections = self._detect_yolo(frame)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")

        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect using Ultralytics YOLO"""
        detections = []

        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        # Extract person detections (class 0 in COCO)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:  # Person class
                        detection = Detection(
                            bbox=box.xyxy[0].cpu().numpy().tolist(),
                            confidence=float(box.conf),
                            class_id=0,
                            class_name="person"
                        )
                        detections.append(detection)

        return detections

    def _detect_opencv_dnn(self, frame: np.ndarray) -> List[Detection]:
        """Detect using OpenCV DNN (fallback)"""
        detections = []

        height, width = frame.shape[:2]

        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Forward pass
        outs = self.net.forward(self.output_layers)

        # Process outputs
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only detect people (class 0 in COCO)
                if class_id == 0 and confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        if len(indices) > 0:
            if hasattr(indices, '__len__') and not isinstance(indices, np.ndarray):
                indices = indices.flatten()

            for i in indices:
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]

                x, y, w, h = boxes[i]
                detection = Detection(
                    bbox=[x, y, x + w, y + h],
                    confidence=confidences[i],
                    class_id=class_ids[i],
                    class_name="person"
                )
                detections.append(detection)

        return detections