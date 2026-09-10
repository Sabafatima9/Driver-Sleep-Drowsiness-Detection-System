"""
Driver Sleep & Drowsiness Detection System (DMS) using MediaPipe and Tkinter.
Detects driver fatigue, micro-sleeps, prolonged eye closure, yawning, and head nodding/slump
in real time for both live webcam streams and recorded driving video files.

Compatible with modern MediaPipe Tasks (mediapipe >= 1.0.0, FaceLandmarker)
and legacy MediaPipe Solutions (mediapipe < 1.0.0, mp.solutions.face_mesh).
"""

import os
import sys
import time
import math
import csv
import threading
import urllib.request
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# Enable High-DPI scaling on Windows if available
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Audio alert support (Windows winsound)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ==============================================================================
# MODEL CONFIGURATION & DOWNLOAD URL
# ==============================================================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"

# 3D Head Pose Model Landmarks
HEAD_POSE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1: Nose tip
    [0.0, -330.0, -65.0],      # 152: Chin
    [-225.0, 170.0, -135.0],   # 33: Left eye left corner
    [225.0, 170.0, -135.0],    # 263: Right eye right corner
    [-150.0, -150.0, -125.0],  # 61: Left mouth corner
    [150.0, -150.0, -125.0]    # 291: Right mouth corner
], dtype=np.float64)
KEY_POSE_INDICES = [1, 152, 33, 263, 61, 291]

# Eye Landmarks for Eye Aspect Ratio (EAR)
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]   # Outer, Top1, Top2, Inner, Bot2, Bot1
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380] # Outer, Top1, Top2, Inner, Bot2, Bot1

# Eye and Mouth Contours
LEFT_EYE_CONTOUR = [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)]
RIGHT_EYE_CONTOUR = [(263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)]
LIPS_CONTOUR = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291)]

# Themes (BGR format for OpenCV)
COLOR_THEMES = {
    "Cyberpunk Neon": {
        "face": (255, 230, 0),      # Cyan
        "eye_open": (0, 255, 128),  # Neon Mint
        "eye_closed": (0, 0, 255),  # Pure Red
        "mouth_norm": (255, 0, 180),# Neon Pink
        "mouth_yawn": (0, 140, 255),# Hot Orange
        "hud_text": (240, 245, 255),
        "danger": (0, 0, 255),
        "warning": (0, 165, 255),
        "safe": (0, 255, 128),
        "badge": "#38bdf8",
    },
    "Matrix Emerald": {
        "face": (80, 255, 80),
        "eye_open": (100, 255, 100),
        "eye_closed": (0, 0, 255),
        "mouth_norm": (50, 220, 50),
        "mouth_yawn": (0, 140, 255),
        "hud_text": (220, 255, 220),
        "danger": (0, 0, 255),
        "warning": (0, 200, 255),
        "safe": (0, 255, 0),
        "badge": "#10b981",
    },
    "Electric Sunset": {
        "face": (0, 180, 255),
        "eye_open": (0, 215, 255),
        "eye_closed": (0, 0, 255),
        "mouth_norm": (180, 50, 255),
        "mouth_yawn": (0, 90, 255),
        "hud_text": (255, 240, 230),
        "danger": (0, 0, 255),
        "warning": (0, 140, 255),
        "safe": (0, 255, 128),
        "badge": "#f59e0b",
    },
    "Sci-Fi Cyan": {
        "face": (255, 220, 0),
        "eye_open": (255, 235, 100),
        "eye_closed": (0, 0, 255),
        "mouth_norm": (255, 170, 0),
        "mouth_yawn": (0, 140, 255),
        "hud_text": (240, 245, 255),
        "danger": (0, 0, 255),
        "warning": (0, 180, 255),
        "safe": (0, 255, 200),
        "badge": "#06b6d4",
    },
}


# ==============================================================================
# SOUND ALARM MANAGER (Asynchronous / Non-blocking)
# ==============================================================================
class AlarmPlayer:
    """Plays audio alerts in a background thread to prevent GUI freezing."""

    def __init__(self):
        self.enabled = True
        self.is_playing = False
        self.last_beep_time = 0.0

    def trigger_alarm(self, freq=1400, duration=350):
        """Triggers alarm buzzer asynchronously."""
        if not self.enabled or not HAS_WINSOUND:
            return

        curr = time.time()
        # Prevent sound overlap/spam (wait at least 0.4s between triggers)
        if curr - self.last_beep_time < 0.4:
            return

        self.last_beep_time = curr

        def _play():
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()


# ==============================================================================
# MEDIAPIPE FACE MESH DETECTOR
# ==============================================================================
class FaceDetector:
    """Wrapper that provides unified 468/478-point face landmarks for drowsiness tracking."""

    def __init__(self, confidence=0.5):
        self.confidence = confidence
        self.use_tasks = False
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.use_tasks = False
            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
            )
        else:
            self.use_tasks = True
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
            if not os.path.exists(model_path):
                print(f"Model file not found. Downloading {MODEL_FILENAME} from Google MediaPipe...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("Model downloaded successfully!")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=self.confidence,
                min_face_presence_confidence=self.confidence,
                min_tracking_confidence=self.confidence,
                output_face_blendshapes=True,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        """Processes a frame and returns 468 landmarks (x, y, z, raw_x, raw_y)."""
        import mediapipe as mp
        h, w, _ = frame_bgr.shape

        # Downsample large 4K frames for fast inference
        scale = 640.0 / max(w, h)
        if scale < 1.0:
            small_w, small_h = int(w * scale), int(h * scale)
            inference_frame = cv2.resize(frame_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            inference_frame = frame_bgr

        if self.use_tasks:
            frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)

            if result and result.face_landmarks and len(result.face_landmarks) > 0:
                pts = []
                for lm in result.face_landmarks[0]:
                    pts.append((int(lm.x * w), int(lm.y * h), lm.z, lm.x, lm.y))
                return pts
        else:
            frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            result = self.detector.process(frame_rgb)

            if result and result.multi_face_landmarks:
                pts = []
                for lm in result.multi_face_landmarks[0].landmark:
                    pts.append((int(lm.x * w), int(lm.y * h), lm.z, lm.x, lm.y))
                return pts

        return None


# ==============================================================================
# FATIGUE & SLEEP ANALYZER ENGINE
# ==============================================================================
class DrowsinessEngine:
    """Evaluates EAR, MAR, Head Nodding, PERCLOS, and triggers safety alarms."""

    def __init__(self, fps=30.0):
        self.fps = fps if fps > 0 else 30.0

        # Configurable thresholds
        self.ear_threshold = 0.20       # Below this = Eye is closed
        self.mar_threshold = 0.50       # Above this = Mouth is wide open / Yawn
        self.sleep_time_threshold = 1.4 # Seconds of closed eyes to trigger CRITICAL SLEEP ALARM
        self.yawn_time_threshold = 1.5  # Seconds of mouth open to trigger YAWN ALARM
        self.head_nod_pitch = -16.0     # Degrees forward drop = Head slump / nodding off
        self.distraction_yaw = 32.0     # Degrees turn away = Looking away from road

        # State counters
        self.closed_eyes_frames = 0
        self.yawn_frames = 0
        self.head_nod_frames = 0
        self.distraction_frames = 0

        self.total_yawns = 0
        self.total_sleep_alarms = 0
        self.total_nod_alarms = 0

        # Rolling history for PERCLOS & Waveforms (last ~3 seconds / 90 frames)
        self.window_size = 90
        self.ear_history = deque(maxlen=self.window_size)
        self.mar_history = deque(maxlen=self.window_size)
        self.eye_closure_binary = deque(maxlen=self.window_size)

        # Incident log
        self.incident_log = []

        # Current live status
        self.ear = 0.32
        self.mar = 0.20
        self.perclos = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.fatigue_score = 0  # 0 - 100%
        self.alert_level = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"
        self.alert_message = "ALERT & ATTENTIVE"

    def reset(self):
        """Resets all metrics and counters."""
        self.closed_eyes_frames = 0
        self.yawn_frames = 0
        self.head_nod_frames = 0
        self.distraction_frames = 0
        self.total_yawns = 0
        self.total_sleep_alarms = 0
        self.total_nod_alarms = 0
        self.ear_history.clear()
        self.mar_history.clear()
        self.eye_closure_binary.clear()
        self.incident_log.clear()
        self.alert_level = "NORMAL"
        self.alert_message = "ALERT & ATTENTIVE"
        self.fatigue_score = 0

    def compute_ear(self, pts, eye_indices):
        """Computes Eye Aspect Ratio (EAR) for an eye."""
        p1 = pts[eye_indices[0]] # Outer
        p2 = pts[eye_indices[1]] # Top 1
        p3 = pts[eye_indices[2]] # Top 2
        p4 = pts[eye_indices[3]] # Inner
        p5 = pts[eye_indices[4]] # Bot 2
        p6 = pts[eye_indices[5]] # Bot 1

        v1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        v2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        h = math.hypot(p1[0] - p4[0], p1[1] - p4[1])

        if h < 1e-5:
            return 0.30
        return (v1 + v2) / (2.0 * h)

    def compute_mar(self, pts):
        """Computes Mouth Aspect Ratio (MAR)."""
        # Upper inner lip: 13, Lower inner lip: 14
        # Left corner: 61, Right corner: 291
        v = math.hypot(pts[13][0] - pts[14][0], pts[13][1] - pts[14][1])
        h = math.hypot(pts[61][0] - pts[291][0], pts[61][1] - pts[291][1])

        if h < 1e-5:
            return 0.10
        return v / h

    def estimate_head_pose(self, pts, img_w, img_h):
        """Estimates 3D Head Orientation (Pitch, Yaw, Roll)."""
        try:
            pts_2d = np.array([pts[idx][:2] for idx in KEY_POSE_INDICES], dtype=np.float64)
            focal_length = float(img_w)
            center = (float(img_w) / 2.0, float(img_h) / 2.0)
            cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(HEAD_POSE_3D, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return 0.0, 0.0, 0.0, None

            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles[0], angles[1], angles[2]

            # 3D Vector projection from nose tip
            axis_len = 50.0
            axis_3d = np.array([[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0], [0.0, -axis_len, 0.0], [0.0, 0.0, -axis_len]], dtype=np.float64)
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_matrix, dist_coeffs)

            axis_points = [(int(p.ravel()[0]), int(p.ravel()[1])) for p in imgpts]
            return pitch, yaw, roll, axis_points
        except Exception:
            return 0.0, 0.0, 0.0, None

    def process_frame(self, pts, frame_idx, current_time_sec, img_w, img_h, alarm_player=None):
        """Performs full fatigue evaluation on facial landmarks."""
        if pts is None:
            self.alert_level = "WARNING"
            self.alert_message = "NO DRIVER FACE DETECTED"
            return None

        # 1. Compute EAR (Average of both eyes)
        ear_left = self.compute_ear(pts, LEFT_EYE_POINTS)
        ear_right = self.compute_ear(pts, RIGHT_EYE_POINTS)
        self.ear = (ear_left + ear_right) / 2.0
        self.ear_history.append(self.ear)

        # 2. Compute MAR (Yawn)
        self.mar = self.compute_mar(pts)
        self.mar_history.append(self.mar)

        # 3. Head Pose Estimation
        self.pitch, self.yaw, self.roll, axis_points = self.estimate_head_pose(pts, img_w, img_h)

        # 4. PERCLOS Calculation (Percentage of Eye Closure over window)
        is_closed = 1 if self.ear < self.ear_threshold else 0
        self.eye_closure_binary.append(is_closed)
        self.perclos = (sum(self.eye_closure_binary) / float(len(self.eye_closure_binary))) * 100.0

        # 5. Eye Closure Duration Tracking
        if self.ear < self.ear_threshold:
            self.closed_eyes_frames += 1
        else:
            self.closed_eyes_frames = 0

        closed_duration = self.closed_eyes_frames / self.fps

        # 6. Yawning Duration Tracking
        if self.mar > self.mar_threshold:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= int(self.yawn_time_threshold * self.fps):
                self.total_yawns += 1
                self._log_incident("YAWN", current_time_sec, f"Yawn duration: {self.yawn_frames / self.fps:.1f}s")
            self.yawn_frames = 0

        # 7. Head Slump / Nodding Forward Tracking
        if self.pitch < self.head_nod_pitch:
            self.head_nod_frames += 1
        else:
            self.head_nod_frames = 0

        head_nod_duration = self.head_nod_frames / self.fps

        # 8. Distraction / Turning Away Tracking
        if abs(self.yaw) > self.distraction_yaw:
            self.distraction_frames += 1
        else:
            self.distraction_frames = 0

        distraction_duration = self.distraction_frames / self.fps

        # 9. Composite Fatigue Danger Score (0 - 100%)
        closure_score = min(closed_duration / self.sleep_time_threshold, 1.0) * 55.0
        perclos_score = min(self.perclos / 25.0, 1.0) * 25.0
        yawn_score = min(self.total_yawns * 7.0, 20.0)
        nod_score = min(head_nod_duration / 1.0, 1.0) * 20.0
        self.fatigue_score = int(min(closure_score + perclos_score + yawn_score + nod_score, 100.0))

        # 10. Alarm Hierarchy & Status Determination
        if closed_duration >= self.sleep_time_threshold:
            self.alert_level = "CRITICAL"
            self.alert_message = f"SLEEP ALERT! EYES CLOSED {closed_duration:.1f}s"
            if alarm_player:
                alarm_player.trigger_alarm(freq=1600, duration=450)
            if self.closed_eyes_frames == int(self.sleep_time_threshold * self.fps):
                self.total_sleep_alarms += 1
                self._log_incident("MICRO_SLEEP", current_time_sec, f"Closed for {closed_duration:.1f}s")

        elif head_nod_duration >= 1.0:
            self.alert_level = "CRITICAL"
            self.alert_message = "HEAD NODDING / SLUMP DETECTED"
            if alarm_player:
                alarm_player.trigger_alarm(freq=1300, duration=350)
            if self.head_nod_frames == int(1.0 * self.fps):
                self.total_nod_alarms += 1
                self._log_incident("HEAD_SLUMP", current_time_sec, f"Pitch: {self.pitch:.1f} deg")

        elif distraction_duration >= 1.8:
            self.alert_level = "WARNING"
            self.alert_message = "DISTRACTION: LOOK AT ROAD!"
            if alarm_player:
                alarm_player.trigger_alarm(freq=900, duration=200)

        elif self.yawn_frames >= int(self.yawn_time_threshold * self.fps):
            self.alert_level = "WARNING"
            self.alert_message = "YAWNING DETECTED - FATIGUE WARNING"

        elif closed_duration >= 0.7 or self.perclos > 18.0:
            self.alert_level = "WARNING"
            self.alert_message = "DROWSY SIGNS DETECTED"

        else:
            self.alert_level = "NORMAL"
            self.alert_message = "ALERT & ATTENTIVE"

        return {
            "ear": self.ear,
            "mar": self.mar,
            "perclos": self.perclos,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "axis_points": axis_points,
            "fatigue_score": self.fatigue_score,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message,
            "closed_duration": closed_duration,
        }

    def _log_incident(self, incident_type, time_sec, detail):
        self.incident_log.append({
            "timestamp": round(time_sec, 2),
            "type": incident_type,
            "detail": detail,
            "ear": round(self.ear, 3),
            "mar": round(self.mar, 3),
            "perclos": round(self.perclos, 1),
        })


# ==============================================================================
# HUD & VISUAL OVERLAY RENDERER
# ==============================================================================
def draw_driver_hud(canvas_img, pts, metrics, engine, theme_name="Cyberpunk Neon", show_mesh=True, show_axes=True, show_waveforms=True):
    """Draws Driver Monitoring System (DMS) HUD, flashing banners, and metrics."""
    theme = COLOR_THEMES.get(theme_name, COLOR_THEMES["Cyberpunk Neon"])
    h, w, _ = canvas_img.shape

    # 1. Draw Eye Contours
    if pts is not None and show_mesh:
        eye_color = theme["eye_closed"] if engine.ear < engine.ear_threshold else theme["eye_open"]
        for s_idx, e_idx in LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR:
            x1, y1 = pts[s_idx][0], pts[s_idx][1]
            x2, y2 = pts[e_idx][0], pts[e_idx][1]
            cv2.line(canvas_img, (x1, y1), (x2, y2), eye_color, 2, cv2.LINE_AA)

        # Draw Mouth Contour
        mouth_color = theme["mouth_yawn"] if engine.mar > engine.mar_threshold else theme["mouth_norm"]
        for s_idx, e_idx in LIPS_CONTOUR:
            x1, y1 = pts[s_idx][0], pts[s_idx][1]
            x2, y2 = pts[e_idx][0], pts[e_idx][1]
            cv2.line(canvas_img, (x1, y1), (x2, y2), mouth_color, 2, cv2.LINE_AA)

        # Forehead target tag
        fx, fy = pts[10][0], pts[10][1]
        cv2.putText(canvas_img, "DRIVER MONITORED", (fx - 45, max(fy - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, theme["face"], 1, cv2.LINE_AA)

    # 2. Draw 3D Head Pose Axes
    if metrics and metrics.get("axis_points") and show_axes:
        axis_pts = metrics["axis_points"]
        origin = axis_pts[0]
        cv2.line(canvas_img, origin, axis_pts[1], (0, 0, 255), 2, cv2.LINE_AA)     # X (Red)
        cv2.line(canvas_img, origin, axis_pts[2], (0, 255, 0), 2, cv2.LINE_AA)     # Y (Green)
        cv2.line(canvas_img, origin, axis_pts[3], (255, 120, 0), 2, cv2.LINE_AA)   # Z (Blue)

    # 3. Emergency Warning Banners (Flashing Red Border & Banner)
    if engine.alert_level == "CRITICAL":
        # Flashing red border around entire frame
        border_thick = 10
        cv2.rectangle(canvas_img, (0, 0), (w, h), (0, 0, 255), border_thick)

        # Flashing high-visibility alert banner at top center
        banner_w, banner_h = min(w - 60, 560), 65
        bx = (w - banner_w) // 2
        by = 25
        # Red background
        cv2.rectangle(canvas_img, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 200), -1)
        cv2.rectangle(canvas_img, (bx, by), (bx + banner_w, by + banner_h), (255, 255, 255), 2)
        # Warning text
        cv2.putText(canvas_img, engine.alert_message, (bx + 20, by + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    elif engine.alert_level == "WARNING":
        banner_w, banner_h = min(w - 60, 520), 55
        bx = (w - banner_w) // 2
        by = 25
        cv2.rectangle(canvas_img, (bx, by), (bx + banner_w, by + banner_h), (0, 140, 255), -1)
        cv2.rectangle(canvas_img, (bx, by), (bx + banner_w, by + banner_h), (255, 255, 255), 2)
        cv2.putText(canvas_img, engine.alert_message, (bx + 20, by + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Live EAR / MAR Waveform Card (Bottom Left)
    if show_waveforms and len(engine.ear_history) > 2:
        card_w, card_h = 260, 80
        cx, cy = 20, h - card_h - 20

        # Dark card backing
        sub = canvas_img[cy:cy + card_h, cx:cx + card_w]
        dark = np.full(sub.shape, 25, dtype=np.uint8)
        canvas_img[cy:cy + card_h, cx:cx + card_w] = cv2.addWeighted(sub, 0.3, dark, 0.7, 0)
        cv2.rectangle(canvas_img, (cx, cy), (cx + card_w, cy + card_h), (60, 75, 95), 1)

        cv2.putText(canvas_img, f"EAR: {engine.ear:.2f} (Thresh: {engine.ear_threshold:.2f})", (cx + 8, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)

        # Draw threshold line
        thresh_y = cy + card_h - int(engine.ear_threshold * (card_h * 1.8))
        cv2.line(canvas_img, (cx, thresh_y), (cx + card_w, thresh_y), (0, 0, 255), 1, cv2.LINE_4)

        # Plot EAR wave
        ear_pts = []
        ears = list(engine.ear_history)
        step_x = card_w / float(engine.window_size)
        for i, val in enumerate(ears):
            px = int(cx + i * step_x)
            py = cy + card_h - int(val * (card_h * 1.8))
            py = max(cy + 22, min(cy + card_h - 2, py))
            ear_pts.append((px, py))

        for i in range(len(ear_pts) - 1):
            val = ears[i]
            col = (0, 0, 255) if val < engine.ear_threshold else (0, 255, 128)
            cv2.line(canvas_img, ear_pts[i], ear_pts[i + 1], col, 2, cv2.LINE_AA)


# ==============================================================================
# TKINTER APPLICATION
# ==============================================================================
class SleepDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Driver Sleep & Drowsiness Detection System — MediaPipe & Tkinter")
        self.root.geometry("1280x820")
        self.root.minsize(1020, 720)
        self.root.configure(bg="#0f172a")

        # Media & Playback State
        self.cap = None
        self.is_camera = True
        self.video_path = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = True
        self.loop_video = True
        self.playback_speed = 1.0

        # UI & Settings
        self.theme_name = tk.StringVar(value="Cyberpunk Neon")
        self.show_mesh = tk.BooleanVar(value=True)
        self.show_axes = tk.BooleanVar(value=True)
        self.show_waveforms = tk.BooleanVar(value=True)
        self.sound_alarm_enabled = tk.BooleanVar(value=True)

        self.ear_thresh_var = tk.DoubleVar(value=0.20)
        self.sleep_time_var = tk.DoubleVar(value=1.4)
        self.yawn_thresh_var = tk.DoubleVar(value=0.50)

        # Output folder for snapshots and reports
        self.snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        self.latest_processed_frame = None

        # Audio Alarm Player
        self.alarm_player = AlarmPlayer()

        # Fatigue Engine & MediaPipe Detector
        try:
            self.detector = FaceDetector(confidence=0.5)
        except Exception as e:
            messagebox.showerror("Detector Error", f"Failed to initialize Face Landmarker:\n{e}")
            sys.exit(1)

        self.engine = DrowsinessEngine(fps=self.fps)

        # Build GUI
        self._create_widgets()

        # Start Camera by default (or user can open recorded video)
        self._start_camera()

        # Clean Exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Video update loop
        self.update_video()

    def _create_widgets(self):
        # -------------------------------------------------------------
        # TOP HEADER
        # -------------------------------------------------------------
        header_frame = tk.Frame(self.root, bg="#1e293b", height=70, padx=20, pady=10)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_box = tk.Frame(header_frame, bg="#1e293b")
        title_box.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_box,
            text="🚗 DRIVER SLEEP & DROWSINESS DETECTOR",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_box,
            text="Real-time Driver Fatigue, Micro-Sleep, Yawn & Head Slump Monitoring with MediaPipe",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        subtitle_lbl.pack(anchor="w")

        # Badges
        badges_box = tk.Frame(header_frame, bg="#1e293b")
        badges_box.pack(side=tk.RIGHT)

        self.badge_status = tk.Label(
            badges_box,
            text="🟢 ALERT",
            font=("Segoe UI", 11, "bold"),
            fg="#10b981",
            bg="#0f172a",
            padx=12,
            pady=6,
        )
        self.badge_status.pack(side=tk.LEFT, padx=4)

        self.badge_ear = tk.Label(
            badges_box,
            text="EAR: 0.32",
            font=("Consolas", 10, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            padx=10,
            pady=6,
        )
        self.badge_ear.pack(side=tk.LEFT, padx=4)

        self.badge_perclos = tk.Label(
            badges_box,
            text="PERCLOS: 0%",
            font=("Consolas", 10, "bold"),
            fg="#f59e0b",
            bg="#0f172a",
            padx=10,
            pady=6,
        )
        self.badge_perclos.pack(side=tk.LEFT, padx=4)

        self.badge_fatigue = tk.Label(
            badges_box,
            text="FATIGUE: 0%",
            font=("Segoe UI", 10, "bold"),
            fg="#10b981",
            bg="#0f172a",
            padx=10,
            pady=6,
        )
        self.badge_fatigue.pack(side=tk.LEFT, padx=4)

        # -------------------------------------------------------------
        # MAIN BODY CONTAINER
        # -------------------------------------------------------------
        body_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=12)
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: Video Viewport & Playback Controls
        left_col = tk.Frame(body_frame, bg="#0f172a")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        self.video_container = tk.Frame(left_col, bg="#1e293b", bd=2, relief=tk.GROOVE)
        self.video_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            self.video_container,
            text="Starting Camera Stream...",
            font=("Segoe UI", 14),
            fg="#94a3b8",
            bg="#020617",
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Playback Controls Bar
        self.controls_bar = tk.Frame(left_col, bg="#1e293b", pady=8, padx=12)
        self.controls_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        # Scrubber Slider
        slider_frame = tk.Frame(self.controls_bar, bg="#1e293b")
        slider_frame.pack(fill=tk.X, pady=(0, 6))

        self.timeline_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_scrub,
        )
        self.timeline_slider.pack(fill=tk.X, expand=True)

        # Button row
        btn_row = tk.Frame(self.controls_bar, bg="#1e293b")
        btn_row.pack(fill=tk.X)

        self.btn_play_pause = tk.Button(
            btn_row,
            text="⏸️ Pause",
            command=self.toggle_play_pause,
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg="#0284c7",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
        )
        self.btn_play_pause.pack(side=tk.LEFT, padx=3)

        self.btn_replay = tk.Button(
            btn_row,
            text="⏮️ Replay",
            command=self.replay_video,
            font=("Segoe UI", 9),
            fg="#ffffff",
            bg="#334155",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
        )
        self.btn_replay.pack(side=tk.LEFT, padx=3)

        self.cb_loop = tk.Checkbutton(
            btn_row,
            text="🔁 Loop Video",
            variable=tk.BooleanVar(value=True),
            command=self.toggle_loop,
            font=("Segoe UI", 9),
            fg="#f1f5f9",
            bg="#1e293b",
            selectcolor="#0f172a",
            cursor="hand2",
        )
        self.cb_loop.pack(side=tk.LEFT, padx=10)

        speed_lbl = tk.Label(btn_row, text="Speed:", font=("Segoe UI", 9), fg="#94a3b8", bg="#1e293b")
        speed_lbl.pack(side=tk.LEFT, padx=(10, 4))

        self.speed_combo = ttk.Combobox(
            btn_row,
            values=["0.5x", "1.0x", "1.5x", "2.0x"],
            width=6,
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.speed_combo.set("1.0x")
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        # Right Column: Control Sidebar
        sidebar = tk.Frame(body_frame, bg="#1e293b", width=340, padx=14, pady=10)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        def make_section(parent, text):
            sec = tk.LabelFrame(
                parent,
                text=f"  {text}  ",
                font=("Segoe UI", 10, "bold"),
                fg="#38bdf8",
                bg="#1e293b",
                bd=1,
                relief=tk.SOLID,
                padx=10,
                pady=6,
            )
            sec.pack(fill=tk.X, pady=4)
            return sec

        # --- Section 1: Video Input Source ---
        sec_source = make_section(sidebar, "Video Stream Source")
        self.lbl_source = tk.Label(
            sec_source, text="Source: Live Webcam 0", font=("Segoe UI", 8), fg="#e2e8f0", bg="#1e293b", anchor="w"
        )
        self.lbl_source.pack(fill=tk.X, pady=(0, 4))

        src_row = tk.Frame(sec_source, bg="#1e293b")
        src_row.pack(fill=tk.X)

        btn_live = tk.Button(
            src_row,
            text="📷 Live Webcam",
            command=self._start_camera,
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#0284c7",
            relief=tk.FLAT,
            padx=6,
            pady=3,
            cursor="hand2",
        )
        btn_live.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        btn_open = tk.Button(
            src_row,
            text="📂 Open Video File",
            command=self.browse_video,
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#334155",
            relief=tk.FLAT,
            padx=6,
            pady=3,
            cursor="hand2",
        )
        btn_open.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))

        # --- Section 2: Sensitivity & Alarms ---
        sec_sens = make_section(sidebar, "Fatigue Sensitivity & Thresholds")

        tk.Label(sec_sens, text="Eye Closure Time (Micro-Sleep):", font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w")
        slider_sleep = ttk.Scale(
            sec_sens, from_=0.6, to=2.5, variable=self.sleep_time_var, orient=tk.HORIZONTAL, command=self.on_threshold_change
        )
        slider_sleep.pack(fill=tk.X, pady=1)
        self.lbl_sleep_val = tk.Label(sec_sens, text=f"Trigger after: {self.sleep_time_var.get():.1f}s", font=("Consolas", 8), fg="#38bdf8", bg="#1e293b")
        self.lbl_sleep_val.pack(anchor="w")

        tk.Label(sec_sens, text="EAR Closure Threshold:", font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(4, 0))
        slider_ear = ttk.Scale(
            sec_sens, from_=0.15, to=0.28, variable=self.ear_thresh_var, orient=tk.HORIZONTAL, command=self.on_threshold_change
        )
        slider_ear.pack(fill=tk.X, pady=1)
        self.lbl_ear_val = tk.Label(sec_sens, text=f"EAR < {self.ear_thresh_var.get():.2f}", font=("Consolas", 8), fg="#38bdf8", bg="#1e293b")
        self.lbl_ear_val.pack(anchor="w")

        # Audio alarm toggle
        cb_sound = tk.Checkbutton(
            sec_sens,
            text="🔊 Audio Alarm Buzzer",
            variable=self.sound_alarm_enabled,
            command=self.toggle_audio_alarm,
            font=("Segoe UI", 8, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            selectcolor="#0f172a",
            cursor="hand2",
        )
        cb_sound.pack(anchor="w", pady=(6, 2))

        # --- Section 3: Visual Overlays & Theme ---
        sec_visual = make_section(sidebar, "Visual Overlays & Theme")
        toggles = [
            ("👁️ Eye & Mouth Mesh Contours", self.show_mesh),
            ("🎯 3D Head Pose Vector Axes", self.show_axes),
            ("📈 Real-Time EAR Oscilloscope", self.show_waveforms),
        ]
        for text, var in toggles:
            cb = tk.Checkbutton(
                sec_visual, text=text, variable=var, font=("Segoe UI", 8), fg="#f1f5f9", bg="#1e293b", selectcolor="#0f172a", cursor="hand2"
            )
            cb.pack(anchor="w", pady=1)

        tk.Label(sec_visual, text="Color Palette:", font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b").pack(anchor="w", pady=(4, 1))
        self.theme_combo = ttk.Combobox(
            sec_visual, textvariable=self.theme_name, values=list(COLOR_THEMES.keys()), state="readonly", font=("Segoe UI", 8)
        )
        self.theme_combo.pack(fill=tk.X, pady=2)

        # --- Section 4: Live Telemetry Card ---
        sec_telem = make_section(sidebar, "Driver Fatigue Telemetry")

        self.lbl_telem_status = tk.Label(
            sec_telem, text="Status: Alert & Awake", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#1e293b", anchor="w"
        )
        self.lbl_telem_status.pack(fill=tk.X, pady=1)

        self.lbl_telem_eyes = tk.Label(
            sec_telem, text="Eyes: Open (Closed: 0.0s)", font=("Consolas", 8), fg="#e2e8f0", bg="#1e293b", anchor="w"
        )
        self.lbl_telem_eyes.pack(fill=tk.X, pady=1)

        self.lbl_telem_yawn = tk.Label(
            sec_telem, text="Mouth: Closed (MAR: 0.00 | Yawns: 0)", font=("Consolas", 8), fg="#e2e8f0", bg="#1e293b", anchor="w"
        )
        self.lbl_telem_yawn.pack(fill=tk.X, pady=1)

        self.lbl_telem_head = tk.Label(
            sec_telem, text="Head Pose: Pitch 0.0° | Yaw 0.0°", font=("Consolas", 8), fg="#38bdf8", bg="#1e293b", anchor="w"
        )
        self.lbl_telem_head.pack(fill=tk.X, pady=1)

        # --- Section 5: Actions & Export ---
        sec_actions = make_section(sidebar, "Actions & Incident Logs")

        btn_act_row = tk.Frame(sec_actions, bg="#1e293b")
        btn_act_row.pack(fill=tk.X, pady=2)

        btn_snap = tk.Button(
            btn_act_row,
            text="📸 Snapshot",
            command=self.save_snapshot,
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#0284c7",
            relief=tk.FLAT,
            padx=6,
            pady=4,
            cursor="hand2",
        )
        btn_snap.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        btn_export = tk.Button(
            btn_act_row,
            text="📄 Export CSV",
            command=self.export_log,
            font=("Segoe UI", 8, "bold"),
            fg="#ffffff",
            bg="#059669",
            relief=tk.FLAT,
            padx=6,
            pady=4,
            cursor="hand2",
        )
        btn_export.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        btn_reset = tk.Button(
            sec_actions,
            text="🔄 Reset Fatigue Metrics",
            command=self.reset_metrics,
            font=("Segoe UI", 8),
            fg="#ef4444",
            bg="#1e293b",
            relief=tk.SOLID,
            bd=1,
            pady=3,
            cursor="hand2",
        )
        btn_reset.pack(fill=tk.X, pady=(4, 0))

        # -------------------------------------------------------------
        # BOTTOM STATUS FOOTER
        # -------------------------------------------------------------
        status_frame = tk.Frame(self.root, bg="#0f172a", height=30, padx=20, pady=4)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.Label(
            status_frame, text="Driver Monitoring System initialized. Active.", font=("Segoe UI", 8), fg="#64748b", bg="#0f172a"
        )
        self.status_text.pack(side=tk.LEFT)

        snap_count = len(os.listdir(self.snapshots_dir)) if os.path.exists(self.snapshots_dir) else 0
        self.lbl_snap_count = tk.Label(
            status_frame, text=f"📁 Snapshots: {snap_count}", font=("Segoe UI", 8), fg="#64748b", bg="#0f172a"
        )
        self.lbl_snap_count.pack(side=tk.RIGHT)

    def _start_camera(self):
        """Initializes live webcam feed."""
        if self.cap is not None:
            self.cap.release()

        self.is_camera = True
        self.video_path = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(0, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.lbl_source.config(text="Source: Live Webcam 0")
            self.status_text.config(text="Connected to live webcam. Monitoring driver.")
            self.timeline_slider.config(state="disabled")
        else:
            self.status_text.config(text="Could not open webcam.")

    def browse_video(self):
        """Opens recorded video file."""
        path = filedialog.askopenfilename(
            title="Select Driving Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
        )
        if path:
            self._load_video(path)

    def _load_video(self, path):
        """Loads video file into player."""
        if self.cap is not None:
            self.cap.release()

        self.is_camera = False
        self.video_path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video:\n{path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.engine.fps = self.fps
        self.engine.reset()

        self.timeline_slider.config(state="normal", to=max(self.total_frames - 1, 1))
        self.timeline_slider.set(0)
        self.lbl_source.config(text=f"File: {os.path.basename(path)}")
        self.status_text.config(text=f"Loaded {os.path.basename(path)} ({self.total_frames} frames)")

    def on_scrub(self, val):
        if not self.is_camera and self.cap is not None:
            target = int(float(val))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_frame_idx = target

    def on_speed_change(self, event=None):
        speed_map = {"0.5x": 0.5, "1.0x": 1.0, "1.5x": 1.5, "2.0x": 2.0}
        self.playback_speed = speed_map.get(self.speed_combo.get(), 1.0)

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.config(text="⏸️ Pause", bg="#0284c7")
        else:
            self.btn_play_pause.config(text="▶️ Play", bg="#10b981")

    def toggle_loop(self):
        self.loop_video = not self.loop_video

    def replay_video(self):
        if not self.is_camera and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.engine.reset()
            self.is_playing = True
            self.btn_play_pause.config(text="⏸️ Pause", bg="#0284c7")

    def on_threshold_change(self, val=None):
        self.engine.ear_threshold = self.ear_thresh_var.get()
        self.engine.sleep_time_threshold = self.sleep_time_var.get()
        self.lbl_ear_val.config(text=f"EAR < {self.engine.ear_threshold:.2f}")
        self.lbl_sleep_val.config(text=f"Trigger after: {self.engine.sleep_time_threshold:.1f}s")

    def toggle_audio_alarm(self):
        self.alarm_player.enabled = self.sound_alarm_enabled.get()

    def reset_metrics(self):
        self.engine.reset()
        self.status_text.config(text="Fatigue metrics and counters reset.")

    def save_snapshot(self):
        if self.latest_processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dms_snapshot_{timestamp}.png"
            filepath = os.path.join(self.snapshots_dir, filename)
            cv2.imwrite(filepath, self.latest_processed_frame)
            snap_count = len(os.listdir(self.snapshots_dir))
            self.lbl_snap_count.config(text=f"📁 Snapshots: {snap_count}")
            self.status_text.config(text=f"📸 Snapshot saved: {filename}")
        else:
            self.status_text.config(text="No frame available.")

    def export_log(self):
        """Exports sleep/fatigue incident log to CSV."""
        if not self.engine.incident_log:
            messagebox.showinfo("Export Info", "No sleep or fatigue incidents recorded yet.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            initialfile=f"dms_incidents_{timestamp}.csv",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp_Sec", "Incident_Type", "Details", "EAR", "MAR", "PERCLOS_Percent"])
                    for row in self.engine.incident_log:
                        writer.writerow([row["timestamp"], row["type"], row["detail"], row["ear"], row["mar"], row["perclos"]])
                messagebox.showinfo("Success", f"Exported incident report to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")

    def _read_and_process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            if not self.is_camera and self.loop_video and self.total_frames > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                return
            else:
                self.is_playing = False
                self.btn_play_pause.config(text="▶️ Play", bg="#10b981")
                return

        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if not self.is_camera else self.current_frame_idx + 1
        h, w, _ = frame.shape
        curr_time_sec = self.current_frame_idx / self.fps if self.fps > 0 else 0.0

        # Detect Face Landmarks
        pts = self.detector.process(frame)

        # Process Fatigue & Drowsiness
        metrics = self.engine.process_frame(
            pts,
            self.current_frame_idx,
            curr_time_sec,
            w,
            h,
            alarm_player=self.alarm_player,
        )

        # Render HUD
        display_frame = frame.copy()
        draw_driver_hud(
            display_frame,
            pts,
            metrics,
            self.engine,
            theme_name=self.theme_name.get(),
            show_mesh=self.show_mesh.get(),
            show_axes=self.show_axes.get(),
            show_waveforms=self.show_waveforms.get(),
        )

        self.latest_processed_frame = display_frame.copy()

        # Update Badges
        if self.engine.alert_level == "CRITICAL":
            self.badge_status.config(text="🔴 SLEEP ALERT!", fg="#ef4444")
        elif self.engine.alert_level == "WARNING":
            self.badge_status.config(text="🟡 WARNING", fg="#f59e0b")
        else:
            self.badge_status.config(text="🟢 ALERT", fg="#10b981")

        self.badge_ear.config(text=f"EAR: {self.engine.ear:.2f}")
        self.badge_perclos.config(text=f"PERCLOS: {self.engine.perclos:.0f}%")

        f_col = "#ef4444" if self.engine.fatigue_score > 60 else ("#f59e0b" if self.engine.fatigue_score > 30 else "#10b981")
        self.badge_fatigue.config(text=f"FATIGUE: {self.engine.fatigue_score}%", fg=f_col)

        # Update Timeline Scrubber if playing recorded video
        if not self.is_camera:
            self.timeline_slider.set(self.current_frame_idx)

        # Update Telemetry labels
        self.lbl_telem_status.config(text=f"Status: {self.engine.alert_message}", fg=f_col)
        closed_dur = self.engine.closed_eyes_frames / self.fps
        self.lbl_telem_eyes.config(text=f"Eyes: {'CLOSED' if self.engine.ear < self.engine.ear_threshold else 'Open'} (Closed: {closed_dur:.1f}s)")
        self.lbl_telem_yawn.config(text=f"Mouth: MAR {self.engine.mar:.2f} | Yawns: {self.engine.total_yawns}")
        self.lbl_telem_head.config(text=f"Head Pose: Pitch {self.engine.pitch:+.1f}° | Yaw {self.engine.yaw:+.1f}°")

        # Convert to RGB and scale proportionally for Tkinter
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        lbl_w = max(self.video_container.winfo_width(), 640)
        lbl_h = max(self.video_container.winfo_height(), 480)

        scale = min(lbl_w / w, lbl_h / h)
        new_w = max(int(w * scale), 320)
        new_h = max(int(h * scale), 240)

        resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")

    def update_video(self):
        if self.is_playing:
            self._read_and_process_frame()

        delay_ms = max(int(1000.0 / (self.fps * self.playback_speed)), 10)
        self.root.after(delay_ms, self.update_video)

    def on_closing(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    root = tk.Tk()
    app = SleepDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
