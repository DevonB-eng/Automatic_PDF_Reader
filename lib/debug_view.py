# ─────────────────────────────────────────────
# debug_view.py
#
# Debug-only 3D orbit view: shows head center, eye spheres, gaze rays,
# the calibrated monitor plane, and any placed gaze markers, from a
# free-orbiting camera you steer with i/j/k/l/[/]/r. This is entirely
# separate from the actual reader — it can be deleted without touching
# tracking or reader logic, which is the point of keeping it isolated
# here since you said it's mostly "set up once, then ignore."
# ─────────────────────────────────────────────
import cv2
import numpy as np
import math
import keyboard

units_per_cm = None  # set from eye_tracking via render_debug_view_orbit's units_per_cm arg

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

# ─────────────────────────────────────────────
# Drawing Functions
# ─────────────────────────────────────────────
class OrbitCamera:
    def __init__(self, yaw=-151.0, pitch=0.0, radius=1500.0, fov_deg=50.0):
        self.yaw = yaw
        self.pitch = pitch
        self.radius = radius
        self.fov_deg = fov_deg
        self.world_frozen = False
        self.pivot_frozen = None

    def update_from_keys(self):
        yaw_step = math.radians(1.5)
        pitch_step = math.radians(1.5)
        zoom_step = 12.0
        changed = False

        if keyboard.is_pressed('j'):
            self.yaw -= yaw_step
            changed = True
        if keyboard.is_pressed('l'):
            self.yaw += yaw_step
            changed = True
        if keyboard.is_pressed('i'):
            self.pitch += pitch_step
            changed = True
        if keyboard.is_pressed('k'):
            self.pitch -= pitch_step
            changed = True
        if keyboard.is_pressed('['):
            self.radius += zoom_step
            changed = True
        if keyboard.is_pressed(']'):
            self.radius = max(80.0, self.radius - zoom_step)
            changed = True

        if keyboard.is_pressed('r'):
            self.yaw = 0.0
            self.pitch = 0.0
            self.radius = 600.0
            changed = True

        self.pitch = max(math.radians(-89), min(math.radians(89), self.pitch))
        self.radius = max(80.0, self.radius)

        if changed:
            print(f"[Orbit Debug] yaw={math.degrees(self.yaw):.2f}°, "
                  f"pitch={math.degrees(self.pitch):.2f}°, radius={self.radius:.2f}")

    def freeze_pivot(self, monitor_center):
        self.world_frozen = True
        self.pivot_frozen = monitor_center.copy()

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    """Duplicated from eye_tracking.py's draw_gaze — kept local here too
    since debug rendering (draw_wireframe_cube below) also needs its own
    copy of small drawing helpers and this file should stay import-free
    of eye_tracking.py to avoid a circular import (eye_tracking imports
    debug_view). Same logic, unchanged."""
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

def draw_wireframe_cube(frame, center, R, size=80):
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

def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
    orbit_cam=None,
    units_per_cm=None,
):
    """Mirrors the original render_debug_view_orbit. The only change:
    orbit state (yaw/pitch/radius/frozen pivot) is now read from an
    OrbitCamera instance passed in as `orbit_cam` instead of module
    globals, and units_per_cm is passed explicitly instead of being
    read from a global set elsewhere."""
    if head_center3d is None:
        return
    if orbit_cam is None:
        orbit_cam = OrbitCamera()

    debug = np.zeros((h, w, 3), dtype=np.uint8)
    head_w = np.asarray(head_center3d, dtype=float)

    if orbit_cam.world_frozen and orbit_cam.pivot_frozen is not None:
        pivot_w = np.asarray(orbit_cam.pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    f_px = _focal_px(w, orbit_cam.fov_deg)
    cam_offset = _rot_y(orbit_cam.yaw) @ (_rot_x(orbit_cam.pitch) @ np.array([0.0, 0.0, orbit_cam.radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])
    fwd = _normalize(pivot_w - cam_pos)
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None:
            return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0)
        b = project_point(P1)
        if a is None or b is None:
            return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0] * ah + l[0] * ah * 0.6), int(p1[1] - v[1] * ah + l[1] * ah * 0.6))
            a2 = (int(p1[0] - v[0] * ah - l[0] * ah * 0.6), int(p1[1] - v[1] * ah - l[1] * ah * 0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:
                parts.append(_normalize(left_dir))
            if right_dir is not None:
                parts.append(_normalize(right_dir))
            if parts:
                combined_dir = _normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs):
                return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    if gaze_markers and monitor_corners is not None:
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0
        v = p3 - p0
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
            and combined_dir is not None
            and sphere_world_l is not None and sphere_world_r is not None):

        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = _normalize(np.asarray(combined_dir, dtype=float))

        C = np.asarray(monitor_center, dtype=float)
        N = _normalize(np.asarray(monitor_normal, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0
                v = p3 - p0
                wv = P - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)
                    b = float(np.dot(wv, v) / v_len2)

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)

                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    help_text = [
        "C = calibrate screen center",
        "J = yaw left",
        "L = yaw right",
        "I = pitch up",
        "K = pitch down",
        "[ = zoom out",
        "] = zoom in",
        "R = reset view",
        "X = add marker",
        "q = quit",
        "F7 = toggle mouse control"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18

    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10

    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)
