# Gaze PDF Reader

A real-time, gaze-controlled PDF reader. A Python eye-tracking backend estimates
where you're looking on screen using nothing but a webcam, and streams that
gaze point to a Flutter desktop app, which renders the PDF and dynamically
magnifies whichever line or paragraph you're currently reading.

## How it works

The system is split into two independent halves connected over a local WebSocket.

**Backend (Python)** — captures webcam frames, runs face/iris landmark
detection, and turns head pose + eye direction into a 2D screen coordinate.
That gaze point is smoothed and broadcast to any connected client at ~60 Hz.

**Frontend (Flutter)** — loads a PDF that's been pre-parsed into layout data
(text blocks, per-line bounding boxes, images), listens for gaze updates over
the socket, and re-renders the page each frame: the paragraph nearest the
gaze point is enlarged in place, everything else stays at normal size.

```bash
Webcam → eye_tracking.py → gaze (x, y) → WebSocket :8765 → main.dart → rendered page
```

## Project structure

| File | Role |
| --- | --- |
| `eye_tracking.py` | Core pipeline: head pose estimation, iris/eye-sphere calibration, gaze-to-screen projection, and the WebSocket server. Entry point (`__main__`) runs the capture loop. |
| `debug_view.py` | Optional 3D orbit view (head, eye spheres, gaze rays, calibrated monitor plane) for visualizing/debugging the tracking geometry. Fully decoupled — safe to remove without touching tracking logic. |
| `reader_display.py` | Standalone OpenCV preview window: applies gaze smoothing/gain and renders a mock text page with the focused line highlighted. Used for testing the smoothing behavior independent of Flutter. |
| `pdf_source.py` | One-time preprocessing script. Parses a PDF with PyMuPDF into `assets/*.json`, preserving per-line bounding boxes, fonts, and embedded images so the Flutter renderer can lay out text exactly as it appears in the source PDF. |
| `main.dart` | Flutter desktop app. Loads the parsed PDF JSON, connects to the gaze WebSocket, and paints the page each frame, magnifying the paragraph under the user's gaze. |

## Setup

### 1. Parse a PDF

```bash
pip install pymupdf
python pdf_source.py
```

Edit the `pdf_path` / `output_json_path` arguments in `pdf_source.py` to point
at your own PDF. This produces a JSON file under `assets/` that the Flutter
app loads at startup.

### 2. Run the eye-tracking backend

```bash
pip install opencv-python numpy mediapipe pyautogui keyboard scipy websockets
python eye_tracking.py
```

This opens a webcam feed and a debug orbit view, and starts a WebSocket
server on `ws://localhost:8765`.

**Calibration controls:**

| Key | Action |
| --- | --- |
| `c` | Lock eye-sphere calibration and calibrate the monitor plane (look at your screen, centered, when pressing) |
| `x` | Add a gaze marker sample (used while refining monitor calibration) |
| `s` | Recenter the gaze cursor |
| `+` / `-` | Adjust horizontal (X) gaze gain |
| `w` / `e` | Adjust vertical (Y) gaze gain |
| `f7` | Toggle mouse control (moves the OS cursor with gaze) |
| `i j k l [ ]` / `r` | Orbit / reset the debug 3D camera |
| `q` | Quit |

### 3. Run the Flutter app

```bash
flutter pub get
flutter run -d windows   # or macos/linux
```

Requires the `web_socket_channel` and `window_manager` packages. The app
connects to the backend's WebSocket and expects the parsed PDF JSON at
`assets/turtles_parsed.json` (update the path/filename in `main.dart` and
`pubspec.yaml` for a different document).

## Notes

- `eye_tracking.py` must be running before (or launched alongside) the
  Flutter app — the app blocks on the WebSocket connection for live gaze data.
- `reader_display.py` can be run/imported on its own with synthetic gaze
  values to tune the smoothing (`GazeSmoother`) without needing a webcam or
  the Flutter app.
- `debug_view.py` is purely diagnostic and has no effect on tracking output;
  disable it if not needed for a small performance gain.
