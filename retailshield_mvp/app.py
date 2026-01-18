#app.py
"""
RetailShield MVP - Main Application
Minimal Viable Product for retail theft detection
"""

import cv2
import time
import logging
import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime

# Import MVP modules
from detector import PersonDetector
from tracker import ObjectTracker
from pose import PoseEstimator
from gesture_logic import GestureAnalyzer
from alert import AlertManager
from video_utils import VideoStream
from engine import ProcessingEngine


class RetailShieldMVP:
    """Main MVP application class"""

    def __init__(self, config):
        self.config = config
        self.running = False
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.detector = None
        self.tracker = None
        self.pose_estimator = None
        self.gesture_analyzer = None
        self.alert_manager = None
        self.video_stream = None
        self.engine = None

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"retailshield_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )

    def initialize_components(self):
        """Initialize all MVP components"""
        self.logger.info("🚀 Initializing RetailShield MVP...")

        try:
            # Initialize detector
            self.detector = PersonDetector(
                model_path=self.config.get('model_path', 'yolov8n.pt'),
                confidence_threshold=self.config.get('confidence_threshold', 0.65)
            )

            # Initialize tracker
            self.tracker = ObjectTracker(
                max_age=30,
                min_hits=3
            )

            # Initialize pose estimator
            self.pose_estimator = PoseEstimator(
                model_path=self.config.get('pose_model_path', 'yolov8n-pose.pt')
            )

            # Initialize gesture analyzer
            self.gesture_analyzer = GestureAnalyzer(
                concealment_threshold=self.config.get('concealment_threshold', 0.7)
            )

            # Initialize alert manager
            self.alert_manager = AlertManager(
                alert_cooldown=30,
                save_clips=True,
                clips_dir="data/clips"
            )

            # Initialize video stream
            self.video_stream = VideoStream(
                source=self.config.get('camera_source', 0),
                width=self.config.get('width', 1280),
                height=self.config.get('height', 720),
                fps=self.config.get('fps', 30)
            )

            # Initialize processing engine
            self.engine = ProcessingEngine(
                detector=self.detector,
                tracker=self.tracker,
                pose_estimator=self.pose_estimator,
                gesture_analyzer=self.gesture_analyzer,
                alert_manager=self.alert_manager
            )

            self.logger.info("✅ All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            return False

    def run(self):
        """Main run loop"""
        self.logger.info("🎬 Starting RetailShield MVP...")

        if not self.initialize_components():
            return False

        self.running = True

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Start video stream
            if not self.video_stream.start():
                self.logger.error("Failed to start video stream")
                return False

            frame_count = 0
            start_time = time.time()

            # Main processing loop
            while self.running:
                # Read frame
                ret, frame = self.video_stream.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break

                # Process frame through engine
                results = self.engine.process_frame(frame, frame_count)

                # Display frame with overlays (optional)
                if self.config.get('show_display', True):
                    display_frame = self.engine.annotate_frame(frame, results)
                    cv2.imshow('RetailShield MVP', display_frame)

                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Quit requested by user")
                        break

                # Update FPS counter
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    self.logger.debug(f"Processing FPS: {fps:.2f}")

            return True

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            return False

        finally:
            self.shutdown()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def shutdown(self):
        """Clean shutdown of all components"""
        self.logger.info("🛑 Shutting down RetailShield MVP...")

        if self.video_stream:
            self.video_stream.release()

        if self.engine:
            self.engine.shutdown()

        cv2.destroyAllWindows()
        self.logger.info("👋 Shutdown complete")

    def run_headless(self):
        """Run without display (for servers)"""
        self.config['show_display'] = False
        return self.run()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RetailShield MVP - Retail Theft Detection')

    parser.add_argument('--camera', type=str, default='0',
                        help='Camera source (0 for webcam, RTSP URL, or video file path)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display window')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to detection model')
    parser.add_argument('--pose-model', type=str, default='yolov8n-pose.pt',
                        help='Path to pose estimation model')
    parser.add_argument('--confidence', type=float, default=0.65,
                        help='Detection confidence threshold')
    parser.add_argument('--concealment-threshold', type=float, default=0.7,
                        help='Concealment detection threshold')
    parser.add_argument('--save-clips', action='store_true',
                        help='Save video clips of incidents')
    parser.add_argument('--clips-dir', type=str, default='data/clips',
                        help='Directory to save video clips')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create configuration dictionary
    config = {
        'camera_source': args.camera,
        'width': args.width,
        'height': args.height,
        'fps': args.fps,
        'model_path': args.model,
        'pose_model_path': args.pose_model,
        'confidence_threshold': args.confidence,
        'concealment_threshold': args.concealment_threshold,
        'save_clips': args.save_clips,
        'clips_dir': args.clips_dir,
        'show_display': not args.headless
    }

    # Create the app
    app = RetailShieldMVP(config)

    # Run the app
    if args.headless:
        success = app.run_headless()
    else:
        success = app.run()

    if success:
        print("✅ RetailShield MVP completed successfully")
        return 0
    else:
        print("❌ RetailShield MVP encountered errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())