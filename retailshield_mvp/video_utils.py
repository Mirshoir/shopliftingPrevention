"""
Video Utilities - Handles video streaming and processing
Supports webcam, RTSP, and video files
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Union
from pathlib import Path


class VideoStream:
    """Video stream handler for multiple sources"""

    def __init__(self, source: Union[str, int] = 0, width: int = 1280,
                 height: int = 720, fps: int = 30):
        self.logger = logging.getLogger(__name__)
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        # Stream state
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Performance tracking
        self.frame_count = 0
        self.start_time = 0
        self.actual_fps = 0

        self.logger.info(f"Video stream initialized for source: {source}")

    def start(self) -> bool:
        """Start the video stream"""
        try:
            # Parse source
            if isinstance(self.source, int) or self.source.isdigit():
                # Webcam
                self.cap = cv2.VideoCapture(int(self.source))
                self.source_type = "webcam"
            elif self.source.startswith(('rtsp://', 'http://', 'rtmp://')):
                # Network stream
                self.cap = cv2.VideoCapture(self.source)
                self.source_type = "network"
            else:
                # Video file
                if Path(self.source).exists():
                    self.cap = cv2.VideoCapture(self.source)
                    self.source_type = "file"
                else:
                    self.logger.error(f"Video file not found: {self.source}")
                    return False

            # Check if stream opened successfully
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False

            # Configure stream properties
            if self.source_type in ["webcam", "network"]:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                # Get actual properties
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                self.logger.info(f"Stream properties: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Start background thread for reading frames
            self.running = True
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()

            # Wait for first frame
            for _ in range(50):  # 50 attempts
                with self.lock:
                    if self.frame is not None:
                        break
                time.sleep(0.01)

            if self.frame is None:
                self.logger.error("Failed to get first frame")
                return False

            self.start_time = time.time()
            self.logger.info(f"Video stream started successfully ({self.source_type})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start video stream: {e}")
            return False

    def _update_frame(self):
        """Background thread to read frames"""
        while self.running:
            try:
                ret, frame = self.cap.read()

                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame
                        self.frame_count += 1

                        # Calculate FPS every 30 frames
                        if self.frame_count % 30 == 0:
                            elapsed = time.time() - self.start_time
                            self.actual_fps = self.frame_count / elapsed

                # Control frame rate for file sources
                if self.source_type == "file" and self.actual_fps > 0:
                    time.sleep(1.0 / self.actual_fps)

            except Exception as e:
                self.logger.error(f"Error reading frame: {e}")
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read the latest frame"""
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def get_properties(self) -> dict:
        """Get stream properties"""
        if self.cap is None:
            return {}

        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'position': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            'actual_fps': self.actual_fps,
            'source_type': self.source_type
        }

    def is_file(self) -> bool:
        """Check if source is a video file"""
        return self.source_type == "file"

    def is_stream(self) -> bool:
        """Check if source is a live stream"""
        return self.source_type in ["webcam", "network"]

    def seek(self, frame_number: int) -> bool:
        """Seek to specific frame (video files only)"""
        if self.cap and self.is_file():
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return False

    def get_frame_number(self) -> int:
        """Get current frame number"""
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def release(self):
        """Release video stream resources"""
        self.logger.info("Releasing video stream...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        self.logger.info("Video stream released")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class VideoWriter:
    """Utility for writing video clips"""

    def __init__(self, output_dir: str = "data/clips", codec: str = "mp4v"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.codec = codec

        self.writer = None
        self.fps = 30
        self.width = 0
        self.height = 0

        self.logger.info(f"Video writer initialized (output: {output_dir})")

    def start_recording(self, width: int, height: int, fps: int = 30,
                        filename: Optional[str] = None) -> str:
        """Start recording a video clip"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clip_{timestamp}.mp4"

        output_path = self.output_dir / filename

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not self.writer.isOpened():
            self.logger.error(f"Failed to open video writer for {output_path}")
            return ""

        self.width = width
        self.height = height
        self.fps = fps

        self.logger.info(f"Started recording: {output_path}")
        return str(output_path)

    def write_frame(self, frame: cv2.Mat):
        """Write a frame to the video"""
        if self.writer and frame is not None:
            # Resize frame if dimensions don't match
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self.writer.write(frame)

    def stop_recording(self):
        """Stop recording and release writer"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info("Stopped recording")

    def create_clip_from_frames(self, frames: list, fps: int = 30,
                                filename: Optional[str] = None) -> str:
        """Create a video clip from a list of frames"""
        if not frames:
            return ""

        # Get dimensions from first frame
        height, width = frames[0].shape[:2]

        # Start recording
        output_path = self.start_recording(width, height, fps, filename)
        if not output_path:
            return ""

        # Write all frames
        for frame in frames:
            self.write_frame(frame)

        # Stop recording
        self.stop_recording()

        return output_path


def test_camera(index: int = 0) -> bool:
    """Test if a camera is working"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False

    # Try to read a frame
    ret, frame = cap.read()
    cap.release()

    return ret and frame is not None


def list_available_cameras(max_check: int = 10) -> list:
    """List all available cameras"""
    available = []

    for i in range(max_check):
        if test_camera(i):
            available.append(i)

    return available


def resize_frame(frame: cv2.Mat, max_width: int = 1280, max_height: int = 720) -> cv2.Mat:
    """Resize frame while maintaining aspect ratio"""
    if frame is None:
        return None

    height, width = frame.shape[:2]

    # Calculate scaling factor
    scale = min(max_width / width, max_height / height, 1.0)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))

    return frame