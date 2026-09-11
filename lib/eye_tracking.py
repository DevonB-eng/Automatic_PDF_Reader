# ─────────────────────────────────────────────
# eye_tracking.py
#
# Core gaze-tracking model: head pose, eye/iris calibration, and the
# virtual monitor plane. Pure computation lives in the classes below —
# no cv2.imshow calls here, so this module can be imported (e.g. for
# testing/tuning) without opening any windows.
#
# The capture loop lives at the bottom, behind
# `if __name__ == "__main__":`, and wires HeadPose / EyeCalibration /
# MonitorPlane together with debug_view.py and reader_display.py.
# ─────────────────────────────────────────────
import cv2
import numpy as np
import math
import threading
import time
from collections import deque

import mediapipe as mp
import pyautogui
import keyboard
from scipy.spatial.transform import Rotation as Rscipy

from debug_view import OrbitCamera, render_debug_view_orbit
from reader_display import GazeSmoother, create_dynamic_text_display

import asyncio
import websockets
import json

# ─────────────────────────────────────────────
# Global Config
# ─────────────────────────────────────────────
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

filter_length = 15
gaze_length = 350

nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

screen_position_file = "screen_position.txt"

def write_screen_position(x, y):
    with open(screen_position_file, 'w') as f:
        f.write(f"{x},{y}\n")

# ─────────────────────────────────────────────
# WebSocket Server
# ─────────────────────────────────────────────
web_gaze_state = {
    "x": CENTER_X, 
    "y": CENTER_Y, 
    "locked": False
}

async def gaze_broadcaster(websocket):
    """Streams the current gaze state to any connected web UI at ~60Hz."""
    try:
        while True:
            await websocket.send(json.dumps(web_gaze_state))
            await asyncio.sleep(0.016)
    except websockets.exceptions.ConnectionClosed:
        pass # UI disconnected

async def run_server():
    """Context manager to start the server and keep it running."""
    async with websockets.serve(gaze_broadcaster, "localhost", 8765):
        await asyncio.Future()  # Keeps the async context alive forever

def start_websocket_server():
    """Runs the asyncio event loop in a separate thread."""
    asyncio.run(run_server())

# ─────────────────────────────────────────────
# Math Helpers 
# ─────────────────────────────────────────────
def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def compute_scale(points_3d):
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    reference_forward = np.array([0, 0, -1])
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_proj /= np.linalg.norm(xz_proj)
    yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
    if avg_direction[0] < 0:
        yaw_rad = -yaw_rad

    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_proj /= np.linalg.norm(yz_proj)
    pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
    if avg_direction[1] > 0:
        pitch_rad = -pitch_rad

    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)

    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = -yaw_deg

    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg

    yawDegrees = 5 * 3
    pitchDegrees = 2.0 * 2.5

    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch

    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

# ─────────────────────────────────────────────
# HeadPose
# ─────────────────────────────────────────────
class HeadPose:
    def __init__(self, indices, color=(0, 255, 0), size=80):
        self.indices = indices
        self.color = color
        self.size = size
        self.R_ref = None  # replaces the old `[None]` container hack

        # Per-frame outputs, populated by update()
        self.center = None
        self.R_final = None
        self.points_3d = None

    def update(self, frame, face_landmarks, w, h):
        """Compute head center + rotation matrix from nose landmarks.
        Mirrors the original compute_and_draw_coordinate_box, including
        its debug drawing (wireframe cube + axes) onto `frame`."""
        points_3d = np.array([
            [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
            for i in self.indices
        ])

        center = np.mean(points_3d, axis=0)
        for i in self.indices:
            x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
            cv2.circle(frame, (x, y), 3, self.color, -1)

        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        r = Rscipy.from_matrix(eigvecs)
        roll, pitch, yaw = r.as_euler('zyx', degrees=False)
        R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

        if self.R_ref is None:
            self.R_ref = R_final.copy()
        else:
            R_ref = self.R_ref
            for i in range(3):
                if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                    R_final[:, i] *= -1

        self._draw_wireframe_cube(frame, center, R_final, self.size)
        axis_length = self.size * 1.2
        axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
        axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

        for i in range(3):
            end_pt = center + axis_dirs[i] * axis_length
            cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

        self.center = center
        self.R_final = R_final
        self.points_3d = points_3d
        return center, R_final, points_3d

    @staticmethod
    def _draw_wireframe_cube(frame, center, R, size=80):
        right = R[:, 0]
        up = -R[:, 1]
        forward = -R[:, 2]

        hw, hh, hd = size * 1, size * 1, size * 1

        def corner(x_sign, y_sign, z_sign):
            return (center + x_sign * hw * right + y_sign * hh * up + z_sign * hd * forward)

        corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
        projected = [(int(pt[0]), int(pt[1])) for pt in corners]

        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for i, j in edges:
            cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

# ─────────────────────────────────────────────
# EyeCalibration
# ─────────────────────────────────────────────
class EyeCalibration:
    def __init__(self, base_radius=20, filter_length=15):
        self.base_radius = base_radius

        self.left_locked = False
        self.left_local_offset = None
        self.left_nose_scale = None

        self.right_locked = False
        self.right_local_offset = None
        self.right_nose_scale = None

        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0

        self.combined_gaze_directions = deque(maxlen=filter_length)

        # Per-frame outputs, populated by update()
        self.sphere_world_l = None
        self.sphere_world_r = None
        self.scaled_radius_l = None
        self.scaled_radius_r = None
        self.iris_3d_left = None
        self.iris_3d_right = None
        self.combined_dir = None  # averaged/filtered gaze direction

    @property
    def is_locked(self):
        return self.left_locked and self.right_locked

    def calibrate(self, head_center, R_final, iris_3d_left, iris_3d_right, nose_points_3d):
        """Lock both eye spheres to their current iris position relative
        to the head. Mirrors the original 'c' key handler."""
        current_nose_scale = compute_scale(nose_points_3d)

        self.left_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = R_final.T @ camera_dir_world
        self.left_local_offset += self.base_radius * camera_dir_local
        self.left_nose_scale = current_nose_scale
        self.left_locked = True

        self.right_local_offset = R_final.T @ (iris_3d_right - head_center)
        self.right_local_offset += self.base_radius * camera_dir_local
        self.right_nose_scale = current_nose_scale
        self.right_locked = True

        sphere_world_l_calib = head_center + R_final @ self.left_local_offset
        sphere_world_r_calib = head_center + R_final @ self.right_local_offset

        left_dir = iris_3d_left - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        if np.linalg.norm(left_dir) > 1e-9:
            left_dir /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9:
            right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint

        return forward_hint, gaze_origin, gaze_dir

    def recenter_screen(self):
        """Mirrors the original 's' key handler: zero out yaw/pitch
        offsets so the current gaze direction maps to screen center."""
        if not self.is_locked:
            return
        left_gaze_dir = self.iris_3d_left - self.sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = self.iris_3d_right - self.sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)

        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0
        )

        self.calibration_offset_yaw = 0 - raw_yaw
        self.calibration_offset_pitch = 0 - raw_pitch
        print(f"[Screen Calibrated] Offset Yaw: {self.calibration_offset_yaw:.2f}, "
              f"Offset Pitch: {self.calibration_offset_pitch:.2f}")

    def update(self, frame, head_center, R_final, nose_points_3d, left_iris, right_iris, w, h):
        """Per-frame update: draw eye markers, compute sphere world
        positions if locked, and (if both locked) the combined gaze
        direction. Mirrors the per-frame body of the original main loop."""
        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        if not self.left_locked:
            cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / self.left_nose_scale if self.left_nose_scale else 1.0
            scaled_offset = self.left_local_offset * scale_ratio
            self.sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(self.sphere_world_l[0]), int(self.sphere_world_l[1])
            self.scaled_radius_l = int(self.base_radius * scale_ratio)
            cv2.circle(frame, (x_sphere_l, y_sphere_l), self.scaled_radius_l, (255, 255, 25), 2)

        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        if not self.right_locked:
            cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / self.right_nose_scale if self.right_nose_scale else 1.0
            scaled_offset_r = self.right_local_offset * scale_ratio_r
            self.sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(self.sphere_world_r[0]), int(self.sphere_world_r[1])
            self.scaled_radius_r = int(self.base_radius * scale_ratio_r)
            cv2.circle(frame, (x_sphere_r, y_sphere_r), self.scaled_radius_r, (25, 255, 255), 2)

        self.iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        self.iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

        self.combined_dir = None
        if self.is_locked:
            draw_gaze(frame, self.sphere_world_l, self.iris_3d_left, self.scaled_radius_l, (55, 255, 0), 130)
            draw_gaze(frame, self.sphere_world_r, self.iris_3d_right, self.scaled_radius_r, (55, 255, 0), 130)

            left_gaze_dir = self.iris_3d_left - self.sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)

            right_gaze_dir = self.iris_3d_right - self.sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)

            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            self.combined_gaze_directions.append(raw_combined_direction)

            avg_combined_direction = np.mean(self.combined_gaze_directions, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            self.combined_dir = avg_combined_direction

        return self.combined_dir

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)
    cv2.line(frame, (int(eye_center[0]), int(eye_center[1])), (int(iris_offset[0]), int(iris_offset[1])), color, 1)

    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)

    cv2.line(frame, (int(iris_offset[0]), int(iris_offset[1])), (int(gaze_endpoint[0]), int(gaze_endpoint[1])), color, 1)

# ─────────────────────────────────────────────
# MonitorPlane
# ─────────────────────────────────────────────
class MonitorPlane:
    def __init__(self):
        self.corners = None
        self.center = None
        self.normal = None
        self.units_per_cm = None
        self.gaze_markers = []

    @property
    def is_calibrated(self):
        return self.corners is not None

    def calibrate(self, head_center, R_final, face_landmarks, w, h, forward_hint=None, gaze_origin=None, gaze_dir=None):
        """Mirrors the original create_monitor_plane function."""
        try:
            lm_chin = face_landmarks[152]
            lm_fore = face_landmarks[10]
            chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
            fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
            face_h_units = np.linalg.norm(fore_w - chin_w)
            upc = face_h_units / 15.0
        except Exception:
            upc = 5.0

        dist_cm = 50.0
        mon_w_cm, mon_h_cm = 60.0, 40.0
        half_w = (mon_w_cm * 0.5) * upc
        half_h = (mon_h_cm * 0.5) * upc

        head_forward = -R_final[:, 2]
        if forward_hint is not None:
            head_forward = forward_hint / np.linalg.norm(forward_hint)

        if gaze_origin is not None and gaze_dir is not None:
            gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)
            plane_point = head_center + head_forward * (50.0 * upc)
            plane_normal = head_forward

            denom = np.dot(plane_normal, gaze_dir)
            if abs(denom) > 1e-6:
                t = np.dot(plane_normal, plane_point - gaze_origin) / denom
                center_w = gaze_origin + t * gaze_dir
            else:
                center_w = head_center + head_forward * (50.0 * upc)
        else:
            center_w = head_center + head_forward * (50.0 * upc)

        world_up = np.array([0, -1, 0], dtype=float)
        head_right = np.cross(world_up, head_forward)
        head_right /= np.linalg.norm(head_right)
        head_up = np.cross(head_forward, head_right)
        head_up /= np.linalg.norm(head_up)

        p0 = center_w - head_right * half_w - head_up * half_h
        p1 = center_w + head_right * half_w - head_up * half_h
        p2 = center_w + head_right * half_w + head_up * half_h
        p3 = center_w - head_right * half_w + head_up * half_h

        normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)

        self.corners = [p0, p1, p2, p3]
        self.center = center_w
        self.normal = normal_w
        self.units_per_cm = upc
        return self.corners, self.center, self.normal, self.units_per_cm

    def add_gaze_marker(self, sphere_world_l, sphere_world_r, combined_dir):
        """Mirrors the original 'x' key handler: intersect the current
        gaze ray with the monitor plane and, if it lands inside the
        plane's bounds, record a marker. Returns True if a marker was
        added, for the caller to log/handle."""
        if not self.is_calibrated:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")
            return False

        D = _normalize(np.asarray(combined_dir, dtype=float))
        O = (sphere_world_l + sphere_world_r) * 0.5
        C = np.asarray(self.center, dtype=float)
        N = _normalize(np.asarray(self.normal, dtype=float))
        denom = float(np.dot(N, D))
        if abs(denom) < 1e-6:
            print("[Marker] Gaze ray parallel to monitor; no marker.")
            return False

        t = float(np.dot(N, (C - O)) / denom)
        if t <= 0.0:
            print("[Marker] Intersection behind/at eye; no marker.")
            return False

        P = O + t * D
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.corners]
        u = p1 - p0
        v = p3 - p0
        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))
        if u_len2 <= 1e-9 or v_len2 <= 1e-9:
            print("[Marker] Monitor dimensions degenerate; no marker.")
            return False

        wv = P - p0
        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            self.gaze_markers.append((a, b))
            print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
            return True
        else:
            print("[Marker] Gaze not on monitor; no marker.")
            return False

# ─────────────────────────────────────────────
# Mouse control
# ─────────────────────────────────────────────
mouse_control_enabled = False
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

def mouse_mover():
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

# ─────────────────────────────────────────────
# Main capture loop
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start the WebSocket server in a background thread
    threading.Thread(target=start_websocket_server, daemon=True).start()
    print("[WebSocket] Server started on ws://localhost:8000")
    
    window_reader = "PDF Reader - Dynamic Text Sizing"
    cv2.namedWindow(window_reader, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_reader, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    threading.Thread(target=mouse_mover, daemon=True).start()

    head_pose = HeadPose(nose_indices, color=(0, 255, 0), size=80)
    eye_calib = EyeCalibration(base_radius=20, filter_length=filter_length)
    monitor = MonitorPlane()
    orbit_cam = OrbitCamera()
    smoother = GazeSmoother()

    prev_focused_line = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        combined_dir = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_iris_idx = 468
            right_iris_idx = 473
            left_iris = face_landmarks[left_iris_idx]
            right_iris = face_landmarks[right_iris_idx]

            head_center, R_final, nose_points_3d = head_pose.update(frame, face_landmarks, w, h)

            combined_dir = eye_calib.update(
                frame, head_center, R_final, nose_points_3d, left_iris, right_iris, w, h
            )

            if eye_calib.is_locked and combined_dir is not None:
                screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                    combined_dir,
                    eye_calib.calibration_offset_yaw,
                    eye_calib.calibration_offset_pitch
                )

                # (Existing code)
                smoother.update(screen_x, screen_y, MONITOR_WIDTH, MONITOR_HEIGHT)
                
                # Update the state for the WebSocket to broadcast
                web_gaze_state["x"] = smoother.x
                web_gaze_state["y"] = smoother.y
                web_gaze_state["locked"] = eye_calib.is_locked

                if mouse_control_enabled:
                    with mouse_lock:
                        mouse_target[0] = screen_x
                        mouse_target[1] = screen_y

                write_screen_position(screen_x, screen_y)

                combined_origin = (eye_calib.sphere_world_l + eye_calib.sphere_world_r) / 2
                combined_target = combined_origin + combined_dir * gaze_length
                cv2.line(
                    frame,
                    (int(combined_origin[0]), int(combined_origin[1])),
                    (int(combined_target[0]), int(combined_target[1])),
                    (255, 255, 10), 3
                )

                text = f"Screen: ({screen_x}, {screen_y})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = (w - text_width) // 2
                color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
                cv2.putText(frame, text, (center_x, 30), font, font_scale, color, thickness)

            for idx, lm in enumerate(face_landmarks):
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

            orbit_cam.update_from_keys()

            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in face_landmarks], dtype=float)

            render_debug_view_orbit(
                h, w,
                head_center3d=head_center,
                sphere_world_l=eye_calib.sphere_world_l if eye_calib.left_locked else None,
                scaled_radius_l=eye_calib.scaled_radius_l if eye_calib.left_locked else None,
                sphere_world_r=eye_calib.sphere_world_r if eye_calib.right_locked else None,
                scaled_radius_r=eye_calib.scaled_radius_r if eye_calib.right_locked else None,
                iris3d_l=eye_calib.iris_3d_left,
                iris3d_r=eye_calib.iris_3d_right,
                left_locked=eye_calib.left_locked,
                right_locked=eye_calib.right_locked,
                landmarks3d=landmarks3d,
                combined_dir=combined_dir,
                gaze_len=5230,
                monitor_corners=monitor.corners,
                monitor_center=monitor.center,
                monitor_normal=monitor.normal,
                gaze_markers=monitor.gaze_markers,
                orbit_cam=orbit_cam,
                units_per_cm=monitor.units_per_cm,
            )

        # ─────────────────────────────────────────────
        # Generate & Show PDF Reader Window
        # ─────────────────────────────────────────────
        gaze_y_val = smoother.y if eye_calib.is_locked else None

        display, focused_line = create_dynamic_text_display(
            MONITOR_WIDTH, MONITOR_HEIGHT, gaze_y_val
        )

        if prev_focused_line is not None and focused_line is not None:
            if abs(focused_line - prev_focused_line) <= 1:
                focused_line = prev_focused_line
        prev_focused_line = focused_line

        cv2.circle(display, (smoother.x, smoother.y), 10, (0, 0, 255), -1)
        cv2.circle(display, (smoother.x, smoother.y), 12, (255, 255, 255), 2)

        info_panel = np.zeros((35, MONITOR_WIDTH, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)
        focus_text = f"Line {focused_line}" if focused_line is not None else "None"
        cv2.putText(info_panel,
                    f"Focus: {focus_text}  |  "
                    f"Gaze Y: {smoother.y}  |  "
                    f"Gain X: {smoother.x_gain:.1f}  Y: {smoother.y_gain:.1f}",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        display[0:35, 0:MONITOR_WIDTH] = info_panel

        cv2.putText(display, "PDF Reader controls: +/- = X gain, w/e = Y gain",
                    (20, MONITOR_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow(window_reader, display)
        cv2.imshow("Integrated Eye Tracking", frame)

        if keyboard.is_pressed('f7'):
            mouse_control_enabled = not mouse_control_enabled
            print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
            time.sleep(0.3)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # PDF Reader Gain Keys
        elif key == ord('+') or key == ord('='):
            smoother.adjust_x_gain(0.1)
        elif key == ord('-'):
            smoother.adjust_x_gain(-0.1)
        elif key == ord('w'):  # Mapped from ']' to avoid zoom conflict
            smoother.adjust_y_gain(0.1)
        elif key == ord('e'):  # Mapped from '[' to avoid zoom conflict
            smoother.adjust_y_gain(-0.1)

        # Tracker Calibrations
        elif key == ord('c') and not eye_calib.is_locked and results.multi_face_landmarks:
            forward_hint, gaze_origin, gaze_dir = eye_calib.calibrate(
                head_center, R_final, eye_calib.iris_3d_left, eye_calib.iris_3d_right, nose_points_3d
            )

            monitor.calibrate(
                head_center, R_final, face_landmarks, w, h,
                forward_hint=forward_hint,
                gaze_origin=gaze_origin,
                gaze_dir=gaze_dir
            )

            orbit_cam.freeze_pivot(monitor.center)
            print("[Debug View] World pivot frozen at monitor center.")
            print(f"[Monitor] units_per_cm={monitor.units_per_cm:.3f}, center={monitor.center}, normal={monitor.normal}")
            print("[Both Spheres Locked] Eye sphere calibration complete.")

        elif key == ord('s') and eye_calib.is_locked:
            eye_calib.recenter_screen()

        elif key == ord('x'):
            if monitor.is_calibrated and eye_calib.is_locked and results.multi_face_landmarks:
                current_nose_scale = compute_scale(nose_points_3d)
                scale_ratio_l = current_nose_scale / eye_calib.left_nose_scale if eye_calib.left_nose_scale else 1.0
                scale_ratio_r = current_nose_scale / eye_calib.right_nose_scale if eye_calib.right_nose_scale else 1.0
                sphere_world_l_now = head_center + R_final @ (eye_calib.left_local_offset * scale_ratio_l)
                sphere_world_r_now = head_center + R_final @ (eye_calib.right_local_offset * scale_ratio_r)

                if combined_dir is not None:
                    D_dir = combined_dir
                else:
                    lg = eye_calib.iris_3d_left - sphere_world_l_now
                    rg = eye_calib.iris_3d_right - sphere_world_r_now
                    if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                        print("[Marker] Gaze direction invalid; try again.")
                        D_dir = None
                    else:
                        lg /= np.linalg.norm(lg)
                        rg /= np.linalg.norm(rg)
                        D_dir = _normalize(lg + rg)

                if D_dir is not None:
                    monitor.add_gaze_marker(sphere_world_l_now, sphere_world_r_now, D_dir)
            else:
                print("[Marker] Monitor/gaze not ready; complete center calibration first.")

    cap.release()
    cv2.destroyAllWindows()