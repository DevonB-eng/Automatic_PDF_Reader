# ─────────────────────────────────────────────
# reader_display.py
#
# renders the text page and enlarges/highlights
# the line the user is currently looking at. Takes a raw gaze point
# in (via GazeSmoother) and has no knowledge of how that point was
# computed — it doesn't import eye_tracking or debug_view.
# ─────────────────────────────────────────────
import cv2
import numpy as np

# ─────────────────────────────────────────────
# GazeSmoother
# ─────────────────────────────────────────────
class GazeSmoother:
    def __init__(self, center_x=None, center_y=None, alpha=0.1, x_gain=0.9, y_gain=0.9):
        self.alpha = alpha
        self.x_gain = x_gain
        self.y_gain = y_gain
        self.x = center_x
        self.y = center_y

    def update(self, raw_screen_x, raw_screen_y, monitor_width, monitor_height):
        """Apply the reader's gain (pull gaze toward center) then
        exponential smoothing. Mirrors the original 'PDF Reader
        Modification Layer' block exactly."""
        if self.x is None:
            self.x = monitor_width // 2
        if self.y is None:
            self.y = monitor_height // 2

        cx, cy = monitor_width // 2, monitor_height // 2
        mod_screen_x = int(cx + (raw_screen_x - cx) * self.x_gain)
        mod_screen_y = int(cy + (raw_screen_y - cy) * self.y_gain)

        self.x = int(self.alpha * mod_screen_x + (1 - self.alpha) * self.x)
        self.y = int(self.alpha * mod_screen_y + (1 - self.alpha) * self.y)
        return self.x, self.y

    def adjust_x_gain(self, delta):
        self.x_gain = round(min(max(self.x_gain + delta, 0.1), 5.0), 1)
        print(f"  X gain -> {self.x_gain}")

    def adjust_y_gain(self, delta):
        self.y_gain = round(min(max(self.y_gain + delta, 0.1), 5.0), 1)
        print(f"  Y gain -> {self.y_gain}")

# ─────────────────────────────────────────────
# Dynamic Text Display Generator (old code for testing view)
# ─────────────────────────────────────────────
def create_dynamic_text_display(screen_w, screen_h, gaze_y):
    display = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 248

    lines = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
        "Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in.",
        "Reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla.",
        "Pariatur. Excepteur sint occaecat cupidatat non proident, sunt in.",
        "Culpa qui officia deserunt mollit anim id est laborum.",
        "",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem.",
        "Accusantium doloremque laudantium, totam rem aperiam, eaque ipsa.",
        "Quae ab illo inventore veritatis et quasi architecto beatae vitae.",
        "Dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit.",
        "Aspernatur aut odit aut fugit, sed quia consequuntur magni dolores.",
        "Eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est.",
        "Qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.",
        "Sed quia non numquam eius modi tempora incidunt ut labore et dolore.",
        "Magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis.",
        "Nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut.",
        "Aliquid ex ea commodi consequatur? Quis autem vel eum iure.",
        "Reprehenderit qui in ea voluptate velit esse quam nihil molestiae.",
        "Consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla.",
        "Pariatur? At vero eos et accusamus et iusto odio dignissimos.",
        "Ducimus qui blanditiis praesentium voluptatum deleniti atque.",
        "Corrupti quos dolores et quas molestias excepturi sint occaecati.",
        "Cupiditate non provident, similique sunt in culpa qui officia.",
        "Deserunt mollitia animi, id est laborum et dolorum fuga. Et harum.",
        "Quidem rerum facilis est et expedita distinctio. Nam libero tempore.",
        "Cum soluta nobis est eligendi optio cumque nihil impedit quo minus.",
        "Id quod maxime placeat facere possimus, omnis voluptas assumenda.",
        "Est, omnis dolor repellendus. Temporibus autem quibusdam et aut.",
        "Officiis debitis aut rerum necessitatibus saepe eveniet ut et.",
        "Voluptates repudiandae sint et molestiae non recusandae. Itaque.",
        "Earum rerum hic tenetur a sapiente delectus, ut aut reiciendis.",
        "Voluptatibus maiores alias consequatur aut perferendis doloribus.",
        "Asperiores repellat.",
    ]

    title_height = 80
    bottom_margin = 40
    available_height = screen_h - title_height - bottom_margin
    margin_x = 60

    base_font_scale = 0.55
    base_line_spacing = 22
    focused_font_scale = 1.1
    focused_line_spacing = 44

    font = cv2.FONT_HERSHEY_SIMPLEX
    test_y = title_height
    line_regions = []

    for i, line_text in enumerate(lines):
        if line_text.strip():
            region_height = base_line_spacing
        else:
            region_height = base_line_spacing // 2

        line_regions.append((int(test_y), int(test_y + region_height), i))
        test_y += region_height

    focused_idx = None
    if gaze_y is not None:
        for start_y, end_y, idx in line_regions:
            if start_y <= gaze_y < end_y:
                focused_idx = idx
                break

        if focused_idx is None and gaze_y >= line_regions[-1][1]:
            focused_idx = len(lines) - 1
        elif focused_idx is None and gaze_y < line_regions[0][0]:
            focused_idx = 0

    num_focused = 0
    num_normal = 0
    if focused_idx is not None:
        for i, line_text in enumerate(lines):
            if abs(i - focused_idx) <= 2:
                if line_text.strip():
                    num_focused += 1
            else:
                if line_text.strip():
                    num_normal += 1
    else:
        num_normal = sum(1 for l in lines if l.strip())

    total_focused_height = num_focused * focused_line_spacing
    remaining = available_height - total_focused_height

    if num_normal > 0:
        normal_spacing = max(12, min(remaining / num_normal, base_line_spacing))
        normal_font_scale = base_font_scale * (normal_spacing / base_line_spacing)
        normal_font_scale = max(0.35, min(normal_font_scale, base_font_scale))
    else:
        normal_spacing = base_line_spacing
        normal_font_scale = base_font_scale

    title = "Lorem Ipsum - Sample Document"
    cv2.putText(display, title, (margin_x, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(display, (margin_x, 60), (screen_w - margin_x, 60), (180, 180, 180), 1)

    y_pos = title_height

    for i, line_text in enumerate(lines):
        is_focused = (focused_idx is not None and abs(i - focused_idx) <= 4)

        if is_focused:
            font_scale = focused_font_scale
            color = (0, 0, 0)
            thickness = 2
            spacing = focused_line_spacing
        else:
            font_scale = normal_font_scale
            color = (80, 80, 80)
            thickness = 1
            spacing = int(normal_spacing)

        if line_text.strip():
            cv2.putText(display, line_text, (margin_x, int(y_pos)), font,
                        float(font_scale), color, thickness, cv2.LINE_AA)

            if is_focused:
                (text_w, text_h), baseline = cv2.getTextSize(line_text, font, float(font_scale), thickness)
                overlay = display.copy()
                cv2.rectangle(overlay,
                              (margin_x - 8, int(y_pos) - text_h - 4),
                              (margin_x + text_w + 8, int(y_pos) + baseline + 4),
                              (0, 240, 255), -1)
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)

                cv2.line(display,
                         (margin_x - 4, int(y_pos) - text_h - 4),
                         (margin_x - 4, int(y_pos) + baseline + 4),
                         (0, 180, 255), 3)

        y_pos += spacing

    page_text = "- 1 -"
    cv2.putText(display, page_text,
                (screen_w // 2 - 25, screen_h - 15),
                font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return display, focused_idx
