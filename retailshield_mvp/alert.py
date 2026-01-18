"""
Alert Manager - Manages alert generation and notifications
Handles alert logic, storage, and notifications
"""

import cv2
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from gesture_logic import GestureResult


@dataclass
class Alert:
    """Alert data class"""
    alert_id: str
    type: str
    confidence: float
    track_id: int
    frame_id: int
    timestamp: str
    description: str


class AlertManager:
    """Manages alert generation, storage, and notifications"""

    def __init__(self, alert_cooldown: int = 30, save_clips: bool = True, clips_dir: str = "data/clips"):
        self.logger = logging.getLogger(__name__)
        self.alert_cooldown = alert_cooldown
        self.save_clips = save_clips
        self.clips_dir = Path(clips_dir)

        # State
        self.active_alerts = {}
        self.alert_history = []
        self.last_alert_time = {}
        self.alert_count = 0

        # Setup directories
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path("data/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Alert manager initialized (cooldown: {alert_cooldown}s)")

    def process_behaviors(self, behaviors: List[GestureResult], frame_id: int) -> List[Alert]:
        """Process detected behaviors and generate alerts"""
        alerts = []

        for behavior in behaviors:
            # Check cooldown for this track
            track_id = getattr(behavior, 'track_id', 0)
            if track_id in self.last_alert_time:
                time_since_last = time.time() - self.last_alert_time[track_id]
                if time_since_last < self.alert_cooldown:
                    continue

            # Create alert
            alert = Alert(
                alert_id=f"alert_{self.alert_count:06d}",
                type=behavior.type,
                confidence=behavior.confidence,
                track_id=track_id,
                frame_id=frame_id,
                timestamp=datetime.now().isoformat(),
                description=behavior.description
            )

            alerts.append(alert)
            self.alert_count += 1

            # Update cooldown
            self.last_alert_time[track_id] = time.time()

        return alerts

    def trigger_alert(self, alert: Alert, frame: Optional[cv2.Mat] = None):
        """Trigger an alert with notifications and logging"""
        self.logger.warning(f"🚨 ALERT {alert.alert_id}: {alert.type} "
                            f"(track {alert.track_id}, confidence: {alert.confidence:.2f})")

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Save alert to log file
        self._save_alert_log(alert)

        # Save frame if provided
        if frame is not None and self.save_clips:
            self._save_alert_frame(alert, frame)

        # Send notification (could be sound, email, etc.)
        self._send_notification(alert)

    def _save_alert_log(self, alert: Alert):
        """Save alert to log file"""
        log_file = self.logs_dir / "alerts.json"

        alert_dict = {
            'alert_id': alert.alert_id,
            'type': alert.type,
            'confidence': alert.confidence,
            'track_id': alert.track_id,
            'frame_id': alert.frame_id,
            'timestamp': alert.timestamp,
            'description': alert.description
        }

        # Load existing alerts or create new list
        alerts_list = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    alerts_list = json.load(f)
            except:
                alerts_list = []

        # Add new alert
        alerts_list.append(alert_dict)

        # Save (keep last 1000 alerts)
        if len(alerts_list) > 1000:
            alerts_list = alerts_list[-1000:]

        with open(log_file, 'w') as f:
            json.dump(alerts_list, f, indent=2)

    def _save_alert_frame(self, alert: Alert, frame: cv2.Mat):
        """Save frame with alert annotation"""
        try:
            # Create directory for this alert
            alert_dir = self.clips_dir / alert.alert_id
            alert_dir.mkdir(exist_ok=True)

            # Draw alert information on frame
            annotated_frame = frame.copy()

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add alert info
            alert_text = f"ALERT: {alert.type} (Track {alert.track_id}, Conf: {alert.confidence:.2f})"
            cv2.putText(annotated_frame, alert_text,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add description
            cv2.putText(annotated_frame, alert.description[:50],
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Save frame
            frame_path = alert_dir / f"frame_{alert.frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)

            self.logger.debug(f"Saved alert frame to {frame_path}")

        except Exception as e:
            self.logger.error(f"Failed to save alert frame: {e}")

    def _send_notification(self, alert: Alert):
        """Send notification for alert"""
        # This is a placeholder - implement actual notification methods

        # Option 1: Console notification
        print(f"\n{'=' * 60}")
        print(f"🚨 RETAILSHIELD ALERT")
        print(f"{'=' * 60}")
        print(f"Type: {alert.type}")
        print(f"Confidence: {alert.confidence:.2f}")
        print(f"Track ID: {alert.track_id}")
        print(f"Time: {alert.timestamp}")
        print(f"Description: {alert.description}")
        print(f"{'=' * 60}\n")

        # Option 2: Sound notification (platform dependent)
        try:
            import platform
            system = platform.system()

            if system == "Darwin":  # macOS
                import os
                os.system('say "Retail Shield Alert detected"')
            elif system == "Linux":
                import os
                os.system('spd-say "Retail Shield Alert detected"')
            elif system == "Windows":
                import winsound
                winsound.Beep(1000, 1000)  # Frequency, Duration
        except:
            pass  # Sound is optional

        # Option 3: Log to file (already done)
        # Option 4: Could add email, SMS, or push notifications here

    def get_active_alerts(self, max_age: int = 300) -> List[Alert]:
        """Get active alerts (not older than max_age seconds)"""
        current_time = time.time()
        active = []

        for alert_id, alert in self.active_alerts.items():
            try:
                alert_time = datetime.fromisoformat(alert.timestamp).timestamp()
                if current_time - alert_time < max_age:
                    active.append(alert)
            except:
                pass

        return active

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts in the last N hours"""
        cutoff_time = time.time() - (hours * 3600)

        # Filter recent alerts
        recent_alerts = []
        for alert in self.alert_history:
            try:
                alert_time = datetime.fromisoformat(alert.timestamp).timestamp()
                if alert_time > cutoff_time:
                    recent_alerts.append(alert)
            except:
                pass

        # Generate summary
        summary = {
            'total_alerts': len(recent_alerts),
            'by_type': {},
            'by_hour': {},
            'average_confidence': 0.0
        }

        if recent_alerts:
            # Count by type
            for alert in recent_alerts:
                summary['by_type'][alert.type] = summary['by_type'].get(alert.type, 0) + 1

            # Group by hour
            for alert in recent_alerts:
                try:
                    dt = datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00'))
                    hour = dt.strftime('%H:00')
                    summary['by_hour'][hour] = summary['by_hour'].get(hour, 0) + 1
                except:
                    pass

            # Average confidence
            total_conf = sum(alert.confidence for alert in recent_alerts)
            summary['average_confidence'] = total_conf / len(recent_alerts)

        return summary

    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Cleanup old alerts from memory"""
        cutoff_time = time.time() - (max_age_hours * 3600)

        # Clean active alerts
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            try:
                alert_time = datetime.fromisoformat(alert.timestamp).timestamp()
                if alert_time < cutoff_time:
                    to_remove.append(alert_id)
            except:
                to_remove.append(alert_id)

        for alert_id in to_remove:
            del self.active_alerts[alert_id]

        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old alerts")