# streamlit_app.py
"""
RetailShield MVP - Streamlit UI
Upload a video and run theft detection pipeline
"""

import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path

# Import MVP modules
from detector import PersonDetector
from tracker import ObjectTracker
from pose import PoseEstimator
from gesture_logic import GestureAnalyzer
from alert import AlertManager
from video_utils import VideoStream
from engine import ProcessingEngine


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="RetailShield MVP",
    layout="wide"
)

st.title("🛡️ RetailShield MVP")
st.caption("Retail theft detection – video-based MVP")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")

confidence_threshold = st.sidebar.slider(
    "Detection confidence",
    min_value=0.3,
    max_value=0.9,
    value=0.65,
    step=0.05
)

concealment_threshold = st.sidebar.slider(
    "Concealment threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.7,
    step=0.05
)

show_keypoints = st.sidebar.checkbox("Show pose keypoints", value=True)
save_clips = st.sidebar.checkbox("Save incident clips", value=True)

# -----------------------------
# Video Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a video",
    type=["mp4", "avi", "mov"]
)

run_button = st.button("▶️ Run Detection", type="primary")

# -----------------------------
# Main Processing
# -----------------------------
if uploaded_file and run_button:

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.success("Video uploaded successfully")

    # Initialize components
    detector = PersonDetector(
        model_path="yolov8n.pt",
        confidence_threshold=confidence_threshold
    )

    tracker = ObjectTracker(max_age=30, min_hits=3)

    pose_estimator = PoseEstimator(
        model_path="yolov8n-pose.pt"
    )

    gesture_analyzer = GestureAnalyzer(
        concealment_threshold=concealment_threshold
    )

    alert_manager = AlertManager(
        alert_cooldown=30,
        save_clips=save_clips,
        clips_dir="data/clips"
    )

    video_stream = VideoStream(
        source=video_path,
        width=1280,
        height=720,
        fps=30
    )

    engine = ProcessingEngine(
        detector=detector,
        tracker=tracker,
        pose_estimator=pose_estimator,
        gesture_analyzer=gesture_analyzer,
        alert_manager=alert_manager
    )

    # Start stream
    if not video_stream.start():
        st.error("Failed to open video")
        st.stop()

    # UI placeholders
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    progress_bar = st.progress(0)

    frame_count = 0
    total_frames = int(video_stream.get_properties().get("frame_count", 0))

    st.info("Processing video...")

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        results = engine.process_frame(frame, frame_count)

        # Annotate frame
        display_frame = engine.annotate_frame(frame, results)

        # Convert BGR → RGB for Streamlit
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(
            display_frame,
            channels="RGB",
            use_container_width=True
        )

        # Show alerts
        if results["alerts"]:
            for alert in results["alerts"]:
                alert_placeholder.warning(
                    f"🚨 {alert.type.upper()} | "
                    f"Confidence: {alert.confidence:.2f} | "
                    f"Track ID: {alert.track_id}"
                )

        frame_count += 1

        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Small delay to avoid UI freeze
        time.sleep(0.01)

    video_stream.release()
    engine.shutdown()

    st.success("✅ Processing completed")

    # Cleanup temp file
    Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video and click **Run Detection**")
