import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import json
import time
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os

# ══════════════════════════════════════════════════════════════════════════════
# THEME — Cyberpunk HUD
# ══════════════════════════════════════════════════════════════════════════════
SETTINGS_FILE = "sv_settings.json"

C = {
    # Backgrounds
    "bg":         "#030508",
    "bg2":        "#080d14",
    "panel":      "#0a1020",
    "card":       "#0c1424",
    "card_hi":    "#101828",
    # Borders
    "border":     "#0f2040",
    "border2":    "#1a3060",
    "border_hot": "#00aaff",
    # Cyan accent (primary)
    "cyan":       "#00e5ff",
    "cyan_dim":   "#00354a",
    "cyan_mid":   "#0077aa",
    # Green accent (active/tracking)
    "green":      "#00ff9d",
    "green_dim":  "#00261a",
    # Red accent (stop/warning)
    "red":        "#ff2d55",
    "red_dim":    "#2a0010",
    # Yellow (calibrating)
    "yellow":     "#ffe600",
    "yellow_dim": "#1a1600",
    # Purple (gesture mapper)
    "purple":     "#bf7fff",
    "purple_dim": "#1a0033",
    # Orange (brow)
    "orange":     "#ff8800",
    # Text
    "text":       "#ccdeff",
    "text2":      "#5a7a9a",
    "text3":      "#203050",
    # Inactive
    "inactive":   "#0a1428",
}

NOISE_ALPHA = 8  # subtle scanline texture opacity

NOSE_TIP   = 1
CALIB_FRAMES = 40
L_EAR_IDX  = [33,  160, 158, 133, 153, 144]
R_EAR_IDX  = [362, 385, 387, 263, 373, 380]
L_BROW_IDX = [105, 159]
R_BROW_IDX = [334, 386]

ACTIONS = [
    "Nothing", "Left Click", "Right Click", "Double Click",
    "Scroll Up", "Scroll Down", "Screenshot", "Pause / Resume Mouse",
]
DEFAULT_MAPPING = {
    "Left Wink":        "Left Click",
    "Right Wink":       "Right Click",
    "Smile":            "Double Click",
    "Raise Eyebrows":   "Scroll Up",
    "Both Eyes Closed": "Pause / Resume Mouse",
}
GESTURE_GLYPHS = {
    "Left Wink": "◑", "Right Wink": "◐",
    "Smile": "⌣", "Raise Eyebrows": "△",
    "Both Eyes Closed": "⊙",
}

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
screen_w, screen_h = pyautogui.size()

_base = python.BaseOptions(model_asset_path="face_landmarker.task")
_opts = vision.FaceLandmarkerOptions(
    base_options=_base, running_mode=vision.RunningMode.VIDEO, num_faces=1)
detector  = vision.FaceLandmarker.create_from_options(_opts)
timestamp = 0


# ══════════════════════════════════════════════════════════════════════════════
# GESTURE + ACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def ear(lm, idx):
    p  = [(lm[i].x, lm[i].y) for i in idx]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    h  = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)

def smile_ratio(lm):
    mw = np.linalg.norm(np.array([lm[61].x, lm[61].y]) - np.array([lm[291].x, lm[291].y]))
    mh = np.linalg.norm(np.array([lm[13].x, lm[13].y]) - np.array([lm[14].x,  lm[14].y]))
    return mw / (mh + 1e-6)

def brow_raise_ratio(lm):
    l = abs(lm[L_BROW_IDX[0]].y - lm[L_BROW_IDX[1]].y)
    r = abs(lm[R_BROW_IDX[0]].y - lm[R_BROW_IDX[1]].y)
    return (l + r) / 2.0

def execute_action(action, toggle_pause_cb):
    if   action == "Left Click":           pyautogui.click(button="left")
    elif action == "Right Click":          pyautogui.click(button="right")
    elif action == "Double Click":         pyautogui.doubleClick()
    elif action == "Scroll Up":            pyautogui.scroll(5)
    elif action == "Scroll Down":          pyautogui.scroll(-5)
    elif action == "Screenshot":
        pyautogui.screenshot().save(f"screenshot_{int(time.time())}.png")
    elif action == "Pause / Resume Mouse": toggle_pause_cb()


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS HELPERS — draw custom widgets on a tk.Canvas
# ══════════════════════════════════════════════════════════════════════════════
def draw_corner_frame(canvas, x1, y1, x2, y2, color, corner=14, width=1):
    """Draw a rectangle with clipped corners (octagon-ish frame)."""
    c = corner
    pts = [
        x1+c, y1,  x2-c, y1,
        x2,   y1+c, x2, y2-c,
        x2-c, y2,  x1+c, y2,
        x1,   y2-c, x1, y1+c,
    ]
    canvas.create_polygon(pts, outline=color, fill="", width=width)

def draw_hud_bar(canvas, x, y, w, h, value, max_val, color, bg_color, label=""):
    """Horizontal fill bar."""
    canvas.create_rectangle(x, y, x+w, y+h, fill=bg_color, outline="")
    filled = int(w * min(value / max_val, 1.0))
    if filled > 0:
        canvas.create_rectangle(x, y, x+filled, y+h, fill=color, outline="")
    if label:
        canvas.create_text(x+4, y+h//2, text=label, anchor="w",
                           fill=color, font=("Courier New", 7, "bold"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class FaceNavApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceNav")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)
        self.root.geometry("1320x820")
        self.root.minsize(1100, 740)

        # State
        self.cap     = cv2.VideoCapture(0)
        self.running = False
        self.paused  = False

        self.calibrating   = True
        self.calib_samples = []
        self.origin_x = self.origin_y = None
        self.smooth_x  = screen_w // 2
        self.smooth_y  = screen_h // 2
        self._calib_pct = 0

        self.sensitivity  = tk.DoubleVar(value=2.5)
        self.smoothing    = tk.DoubleVar(value=0.2)
        self.eye_thresh   = tk.DoubleVar(value=0.20)
        self.smile_thresh = tk.DoubleVar(value=0.30)
        self.brow_thresh  = tk.DoubleVar(value=0.06)
        self.mapping      = {g: tk.StringVar(value=a) for g, a in DEFAULT_MAPPING.items()}
        self.cooldowns    = {g: 0 for g in DEFAULT_MAPPING}

        self.face_detected  = False
        self.cur_x = self.cur_y = 0
        self.live  = {"L EAR": 0.0, "R EAR": 0.0, "SMILE": 0.0, "BROW": 0.0}
        self.gesture_active = {g: False for g in DEFAULT_MAPPING}

        # Toast
        self._toast_id = None

        self._load_settings()
        self._setup_combobox_style()
        self._build_ui()
        self._update_frame()
        self._animate_header()

    # ── Settings ───────────────────────────────────────────────────────────────
    def _load_settings(self):
        if not os.path.exists(SETTINGS_FILE): return
        try:
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
            self.sensitivity.set(s.get("sensitivity",  2.5))
            self.smoothing.set(s.get("smoothing",      0.2))
            self.eye_thresh.set(s.get("eye_thresh",    0.20))
            self.smile_thresh.set(s.get("smile_thresh",0.30))
            self.brow_thresh.set(s.get("brow_thresh",  0.06))
            for g, a in s.get("mapping", {}).items():
                if g in self.mapping: self.mapping[g].set(a)
        except Exception: pass

    def _save_settings(self):
        s = {
            "sensitivity": self.sensitivity.get(), "smoothing": self.smoothing.get(),
            "eye_thresh": self.eye_thresh.get(), "smile_thresh": self.smile_thresh.get(),
            "brow_thresh": self.brow_thresh.get(),
            "mapping": {g: v.get() for g, v in self.mapping.items()},
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f, indent=2)
        self._toast("SETTINGS SAVED", C["green"])

    def _setup_combobox_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("FN.TCombobox",
            fieldbackground=C["card_hi"], background=C["card_hi"],
            foreground=C["cyan"], selectbackground=C["cyan_dim"],
            selectforeground=C["cyan"], arrowcolor=C["cyan"],
            bordercolor=C["border2"], padding=5,
            font=("Courier New", 9))
        s.map("FN.TCombobox",
            fieldbackground=[("readonly", C["card_hi"])],
            foreground=[("readonly", C["cyan"])],
            bordercolor=[("focus", C["cyan"])])

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── HEADER BAR ─────────────────────────────────────────────────────────
        self.header = tk.Canvas(self.root, bg=C["panel"],
                                height=58, highlightthickness=0)
        self.header.pack(fill="x")

        # Top accent line (animated)
        self.header.create_rectangle(0, 0, 2000, 2,
                                     fill=C["cyan"], outline="", tags="accentline")

        # Logo
        self.header.create_text(24, 29, text="◈",
                                font=("Segoe UI Symbol", 20),
                                fill=C["cyan"], anchor="w", tags="logo")
        self.header.create_text(54, 24, text="FACE",
                                font=("Courier New", 16, "bold"),
                                fill=C["text"], anchor="w")
        self.header.create_text(110, 24, text="NAV",
                                font=("Courier New", 16, "bold"),
                                fill=C["cyan"], anchor="w")
        self.header.create_text(54, 42, text="biometric cursor system  v3.0",
                                font=("Courier New", 8),
                                fill=C["text3"], anchor="w")

        # Status badge (right side)
        self.hdr_status_bg = self.header.create_rectangle(
            0, 0, 0, 0, fill=C["inactive"], outline=C["border2"], width=1, tags="status_bg")
        self.hdr_status_dot = self.header.create_text(
            0, 0, text="⬤", font=("Courier New", 9),
            fill=C["text3"], tags="status_dot")
        self.hdr_status_txt = self.header.create_text(
            0, 0, text="IDLE", font=("Courier New", 11, "bold"),
            fill=C["text3"], tags="status_txt")
        self.header.bind("<Configure>", self._reposition_status)

        # Separator
        tk.Frame(self.root, bg=C["border2"], height=1).pack(fill="x")

        # ── BODY ───────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # ────────────────────────────────────────────────────────────────────
        # LEFT COLUMN — camera + gesture activity + live data
        # ────────────────────────────────────────────────────────────────────
        left = tk.Frame(body, bg=C["bg"])
        left.pack(side="left", fill="y", padx=18, pady=18)

        # Camera widget
        cam_card = tk.Frame(left, bg=C["border_hot"], padx=1, pady=1)
        cam_card.pack()
        cam_body = tk.Frame(cam_card, bg=C["panel"])
        cam_body.pack()

        # Camera title strip
        cam_strip = tk.Frame(cam_body, bg=C["card"], height=30)
        cam_strip.pack(fill="x")
        cam_strip.pack_propagate(False)
        tk.Label(cam_strip, text="  SYS::CAM-0   INPUT FEED",
                 font=("Courier New", 8, "bold"),
                 bg=C["card"], fg=C["text2"]).pack(side="left", pady=6)
        self.face_badge = tk.Label(cam_strip, text=" NO FACE DETECTED ",
                                   font=("Courier New", 7, "bold"),
                                   bg=C["red_dim"], fg=C["red"],
                                   padx=6)
        self.face_badge.pack(side="right", padx=8, pady=6)

        self.cam_label = tk.Label(cam_body, bg="#000000",
                                  width=480, height=360)
        self.cam_label.pack()

        # Calibration progress bar
        prog_bg = tk.Frame(cam_body, bg=C["border"], height=4)
        prog_bg.pack(fill="x")
        self.calib_fill = tk.Frame(prog_bg, bg=C["yellow"], height=4, width=0)
        self.calib_fill.place(x=0, y=0, relheight=1)

        # Calibration status
        self.calib_status = tk.Label(cam_body,
            text="  ◌  CALIBRATING  —  HOLD STILL",
            font=("Courier New", 9, "bold"),
            bg=C["panel"], fg=C["yellow"],
            anchor="w", pady=6)
        self.calib_status.pack(fill="x")

        # ── GESTURE ACTIVITY ───────────────────────────────────────────────────
        self._hud_label(left, "GESTURE ACTIVITY", top=14)

        gest_outer = tk.Frame(left, bg=C["card"],
                              highlightbackground=C["border2"],
                              highlightthickness=1)
        gest_outer.pack(fill="x")
        gest_inner = tk.Frame(gest_outer, bg=C["card"], pady=8, padx=8)
        gest_inner.pack(fill="x")

        self.gest_indicators = {}
        short = {"Left Wink": "L-WINK", "Right Wink": "R-WINK",
                 "Smile": "SMILE", "Raise Eyebrows": "BROWS↑",
                 "Both Eyes Closed": "BOTH-SHUT"}
        for g in DEFAULT_MAPPING:
            pill = tk.Frame(gest_inner, bg=C["inactive"],
                            highlightbackground=C["border"],
                            highlightthickness=1,
                            padx=8, pady=5)
            pill.pack(side="left", padx=3)
            g_icon = tk.Label(pill, text=GESTURE_GLYPHS[g],
                              font=("Courier New", 13),
                              bg=C["inactive"], fg=C["text3"])
            g_icon.pack(side="left")
            g_lbl = tk.Label(pill, text=f" {short[g]}",
                             font=("Courier New", 7, "bold"),
                             bg=C["inactive"], fg=C["text3"])
            g_lbl.pack(side="left")
            self.gest_indicators[g] = (pill, g_icon, g_lbl)

        # ── LIVE DATA GRID ─────────────────────────────────────────────────────
        self._hud_label(left, "BIOMETRIC READOUT", top=14)

        data_grid = tk.Frame(left, bg=C["card"],
                             highlightbackground=C["border2"],
                             highlightthickness=1)
        data_grid.pack(fill="x")
        inner_grid = tk.Frame(data_grid, bg=C["card"], padx=10, pady=8)
        inner_grid.pack(fill="x")

        self.live_vars = {}
        cells = [
            ("L-EYE", "l_ear", C["cyan"]),
            ("R-EYE", "r_ear", C["cyan"]),
            ("SMILE", "smile", C["purple"]),
            ("BROW",  "brow",  C["yellow"]),
            ("CUR-X", "cur_x", C["green"]),
            ("CUR-Y", "cur_y", C["green"]),
        ]
        for i, (label, key, color) in enumerate(cells):
            col = i % 3
            row_i = i // 3
            cell = tk.Frame(inner_grid, bg=C["bg2"],
                            highlightbackground=C["border"],
                            highlightthickness=1,
                            padx=8, pady=6)
            cell.grid(row=row_i, column=col, padx=4, pady=3, sticky="ew")
            tk.Label(cell, text=label, font=("Courier New", 7, "bold"),
                     bg=C["bg2"], fg=C["text3"]).pack(anchor="w")
            v = tk.StringVar(value="—")
            self.live_vars[key] = v
            tk.Label(cell, textvariable=v, font=("Courier New", 14, "bold"),
                     bg=C["bg2"], fg=color, width=7, anchor="w").pack()

        # ────────────────────────────────────────────────────────────────────
        # CENTER COLUMN — gesture mapper + controls
        # ────────────────────────────────────────────────────────────────────
        center = tk.Frame(body, bg=C["bg"], width=310)
        center.pack(side="left", fill="y", padx=(0, 14), pady=18)
        center.pack_propagate(False)

        self._hud_label(center, "GESTURE → ACTION BIND")

        for i, (gesture, var) in enumerate(self.mapping.items()):
            row_bg = C["card"] if i % 2 == 0 else C["card_hi"]
            row = tk.Frame(center, bg=row_bg,
                           highlightbackground=C["border2"],
                           highlightthickness=1)
            row.pack(fill="x", pady=2)
            ri = tk.Frame(row, bg=row_bg, pady=9, padx=10)
            ri.pack(fill="x")
            lf = tk.Frame(ri, bg=row_bg)
            lf.pack(side="left", expand=True, fill="x")
            tk.Label(lf, text=GESTURE_GLYPHS[gesture],
                     font=("Courier New", 15),
                     bg=row_bg, fg=C["cyan"]).pack(side="left")
            tk.Label(lf, text=f"  {gesture.upper()}",
                     font=("Courier New", 8, "bold"),
                     bg=row_bg, fg=C["text2"]).pack(side="left")
            ttk.Combobox(ri, textvariable=var, values=ACTIONS,
                         state="readonly", width=15,
                         style="FN.TCombobox").pack(side="right")

        # Controls
        self._hud_label(center, "SYSTEM CONTROLS", top=18)

        self.btn_start = self._hud_btn(center,
            "▶   ENGAGE FACE MOUSE",
            self._toggle_tracking, C["green"], C["green_dim"])
        self._hud_btn(center, "⟳   RECALIBRATE ORIGIN",
                      self._recalibrate, C["cyan"], C["cyan_dim"])
        self._hud_btn(center, "↓   COMMIT SETTINGS",
                      self._save_settings, C["purple"], C["purple_dim"])

        # ────────────────────────────────────────────────────────────────────
        # RIGHT COLUMN — sliders
        # ────────────────────────────────────────────────────────────────────
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True,
                   padx=(0, 18), pady=18)

        self._hud_label(right, "DETECTION PARAMETERS")

        sliders = [
            ("EYE BLINK THRESHOLD",  self.eye_thresh,   0.05, 0.40, C["cyan"],
             "EAR value below which eye is considered closed"),
            ("SMILE SENSITIVITY",    self.smile_thresh,  0.10, 0.60, C["purple"],
             "Mouth width/height ratio required to detect a smile"),
            ("BROW RAISE THRESHOLD", self.brow_thresh,   0.02, 0.15, C["yellow"],
             "Vertical brow-to-eye distance for eyebrow raise"),
            ("CURSOR SENSITIVITY",   self.sensitivity,   1.0,  8.0,  C["green"],
             "Amplification of nose movement to screen movement"),
            ("MOVEMENT SMOOTHING",   self.smoothing,     0.05, 0.5,  C["orange"],
             "Lerp factor — lower = smoother, higher = snappier"),
        ]

        for label, var, lo, hi, color, hint in sliders:
            card = tk.Frame(right, bg=C["card"],
                            highlightbackground=C["border2"],
                            highlightthickness=1)
            card.pack(fill="x", pady=5)
            ci = tk.Frame(card, bg=C["card"], padx=14, pady=10)
            ci.pack(fill="x")

            # Header row
            hrow = tk.Frame(ci, bg=C["card"])
            hrow.pack(fill="x")

            # Colored side marker
            tk.Frame(hrow, bg=color, width=3, height=18).pack(side="left")
            tk.Label(hrow, text=f"  {label}",
                     font=("Courier New", 9, "bold"),
                     bg=C["card"], fg=C["text"]).pack(side="left")

            val_var = tk.StringVar()
            val_lbl = tk.Label(hrow, textvariable=val_var,
                               font=("Courier New", 13, "bold"),
                               bg=C["card"], fg=color,
                               width=6, anchor="e")
            val_lbl.pack(side="right")

            # Hint
            tk.Label(ci, text=hint,
                     font=("Courier New", 7),
                     bg=C["card"], fg="white",
                     anchor="w").pack(fill="x", pady=(3, 6))

            # Slider + range labels
            srow = tk.Frame(ci, bg=C["card"])
            srow.pack(fill="x")
            tk.Label(srow, text=f"{lo:.2f}",
                     font=("Courier New", 8),
                     bg=C["card"], fg=C["text3"]).pack(side="left")
            sl = tk.Scale(srow, variable=var,
                          from_=lo, to=hi, resolution=0.01,
                          orient="horizontal",
                          bg=C["card"], fg=color,
                          troughcolor=C["bg2"],
                          activebackground=color,
                          highlightthickness=0, bd=0,
                          length=240, showvalue=False)
            sl.pack(side="left", padx=(6, 6))
            tk.Label(srow, text=f"{hi:.2f}",
                     font=("Courier New", 8),
                     bg=C["card"], fg=C["text3"]).pack(side="left")

            def _trace(v=var, vv=val_var):
                def cb(*_): vv.set(f"{v.get():.2f}")
                v.trace_add("write", cb); cb()
            _trace()

        # ── Toast ──────────────────────────────────────────────────────────────
        self.toast_var = tk.StringVar(value="")
        self.toast_lbl = tk.Label(self.root, textvariable=self.toast_var,
                                  font=("Courier New", 9, "bold"),
                                  bg=C["green_dim"], fg=C["green"],
                                  padx=18, pady=8,
                                  highlightbackground=C["green"],
                                  highlightthickness=1)

    # ── UI helpers ─────────────────────────────────────────────────────────────
    def _hud_label(self, parent, text, top=0):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", pady=(top, 6))
        # Left tick mark
        tk.Label(f, text="▸", font=("Courier New", 9),
                 bg=C["bg"], fg=C["cyan"]).pack(side="left")
        tk.Label(f, text=f" {text}",
                 font=("Courier New", 8, "bold"),
                 bg=C["bg"], fg=C["text2"]).pack(side="left")
        tk.Frame(f, bg=C["border2"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=5)

    def _hud_btn(self, parent, text, cmd, fg, bg):
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Courier New", 9, "bold"),
                        bg=bg, fg=fg,
                        activebackground=C["card_hi"],
                        activeforeground=fg,
                        relief="flat", bd=0,
                        padx=14, pady=10,
                        cursor="hand2", anchor="w")
        btn.pack(fill="x", pady=3)

        # Hover glow via border highlight
        def on_enter(e, b=btn, c=fg):
            b.config(highlightbackground=c, highlightthickness=1)
        def on_leave(e, b=btn):
            b.config(highlightthickness=0)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _reposition_status(self, event=None):
        w = self.header.winfo_width()
        if w < 10: return
        x1, y1, x2, y2 = w - 180, 12, w - 12, 46
        self.header.coords("status_bg", x1, y1, x2, y2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.header.coords("status_dot", x1 + 18, cy)
        self.header.coords("status_txt", x1 + 34, cy)

    # ── Animated header accent line pulse ─────────────────────────────────────
    def _animate_header(self):
        t   = time.time()
        # Pulse brightness of the accent line
        val = int(120 + 80 * abs(((t * 0.8) % 1.0) - 0.5) * 2)
        val = min(255, max(60, val))
        hex_c = f"#00{val:02x}ff" if val < 230 else "#00e5ff"
        try:
            self.header.itemconfig("accentline", fill=hex_c)
        except Exception:
            pass
        self.root.after(60, self._animate_header)

    # ══════════════════════════════════════════════════════════════════════════
    # WEBCAM LOOP
    # ══════════════════════════════════════════════════════════════════════════
    def _update_frame(self):
        global timestamp
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            fh, fw, _ = frame.shape
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = detector.detect_for_video(mp_image, timestamp)
            timestamp += 33

            self.face_detected = bool(results.face_landmarks)
            if self.face_detected:
                self.face_badge.config(text=" FACE LOCK  ●",
                                       bg=C["green_dim"], fg=C["green"])
            else:
                self.face_badge.config(text=" NO FACE DETECTED ",
                                       bg=C["red_dim"], fg=C["red"])

            if results.face_landmarks:
                lm = results.face_landmarks[0]
                nx = lm[NOSE_TIP].x * fw
                ny = lm[NOSE_TIP].y * fh

                # Calibration
                if self.calibrating:
                    self.calib_samples.append((nx, ny))
                    pct = int(len(self.calib_samples) / CALIB_FRAMES * 100)
                    self._calib_pct = pct
                    self.calib_fill.config(width=int(480 * pct / 100))
                    self.calib_status.config(
                        text=f"  ◌  CALIBRATING  {pct}%  —  HOLD STILL",
                        fg=C["yellow"])
                    cv2.putText(frame, f"CALIBRATING  {pct}%",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 229, 255), 2)
                    if len(self.calib_samples) >= CALIB_FRAMES:
                        self.origin_x = np.mean([s[0] for s in self.calib_samples])
                        self.origin_y = np.mean([s[1] for s in self.calib_samples])
                        self.calibrating = False
                        self.calib_fill.config(bg=C["green"])
                        self.calib_status.config(
                            text="  ✓  ORIGIN LOCKED  —  READY",
                            fg=C["green"])
                else:
                    l_ear_v = ear(lm, L_EAR_IDX)
                    r_ear_v = ear(lm, R_EAR_IDX)
                    smile_v = smile_ratio(lm)
                    brow_v  = brow_raise_ratio(lm)
                    self.live.update({"L EAR": l_ear_v, "R EAR": r_ear_v,
                                      "SMILE": smile_v,  "BROW":  brow_v})

                    et = self.eye_thresh.get()
                    st = self.smile_thresh.get()
                    bt = self.brow_thresh.get()
                    l_closed  = l_ear_v < et
                    r_closed  = r_ear_v < et
                    both_shut = l_closed and r_closed

                    gestures = {
                        "Left Wink":        l_closed and not both_shut,
                        "Right Wink":       r_closed and not both_shut,
                        "Smile":            smile_v > st,
                        "Raise Eyebrows":   brow_v  > bt,
                        "Both Eyes Closed": both_shut,
                    }
                    for g, active in gestures.items():
                        self.gesture_active[g] = active
                        if active and self.cooldowns[g] == 0 and self.running:
                            execute_action(self.mapping[g].get(), self._toggle_pause)
                            self.cooldowns[g] = 25
                        if self.cooldowns[g] > 0:
                            self.cooldowns[g] -= 1

                    # Cursor movement
                    if self.running and not self.paused and not both_shut:
                        dx   = nx - self.origin_x
                        dy   = ny - self.origin_y
                        sens = self.sensitivity.get()
                        tx   = np.clip(screen_w/2 + dx*sens*(screen_w/fw), 0, screen_w-1)
                        ty   = np.clip(screen_h/2 + dy*sens*(screen_h/fh), 0, screen_h-1)
                        sm   = self.smoothing.get()
                        self.smooth_x += sm * (tx - self.smooth_x)
                        self.smooth_y += sm * (ty - self.smooth_y)
                        self.cur_x = int(self.smooth_x)
                        self.cur_y = int(self.smooth_y)
                        pyautogui.moveTo(self.cur_x, self.cur_y)

                    # Frame overlays
                    if self.origin_x:
                        # Origin crosshair
                        ox, oy = int(self.origin_x), int(self.origin_y)
                        cv2.line(frame, (ox-10, oy), (ox+10, oy), (255, 130, 0), 1)
                        cv2.line(frame, (ox, oy-10), (ox, oy+10), (255, 130, 0), 1)
                        # Nose dot + trail
                        cv2.circle(frame, (int(nx), int(ny)), 7, (0, 229, 255), -1)
                        cv2.line(frame, (ox, oy), (int(nx), int(ny)), (0, 255, 157), 1)

                    # EAR bars (bottom of frame)
                    for val, lbl, thresh, bx in [
                            (l_ear_v, "L", et, 8),
                            (r_ear_v, "R", et, 84)]:
                        bar = int(np.clip(val * 210, 0, fw - bx - 4))
                        clr = (80, 40, 255) if val < thresh else (0, 255, 157)
                        cv2.rectangle(frame, (bx, fh-20), (bx+bar, fh-10), clr, -1)
                        cv2.putText(frame, lbl, (bx, fh-24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)

                    # HUD corner brackets
                    margin, length, thick = 8, 18, 2
                    clr_hud = (0, 229, 255)
                    for (x, y, dx, dy) in [
                            (margin, margin, 1, 1),
                            (fw-margin, margin, -1, 1),
                            (margin, fh-margin, 1, -1),
                            (fw-margin, fh-margin, -1, -1)]:
                        cv2.line(frame, (x, y), (x+dx*length, y), clr_hud, thick)
                        cv2.line(frame, (x, y), (x, y+dy*length), clr_hud, thick)

                    if self.paused and self.running:
                        ov = frame.copy()
                        cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 0), -1)
                        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
                        cv2.putText(frame, "// PAUSED //",
                                    (fw//2 - 95, fh//2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 229, 255), 2)

            display = cv2.resize(frame, (480, 360))
            img     = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            imgtk   = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        self._update_live()
        self.root.after(16, self._update_frame)

    # ── Live readout ───────────────────────────────────────────────────────────
    def _update_live(self):
        self.live_vars["l_ear"].set(f"{self.live['L EAR']:.3f}")
        self.live_vars["r_ear"].set(f"{self.live['R EAR']:.3f}")
        self.live_vars["smile"].set(f"{self.live['SMILE']:.2f}")
        self.live_vars["brow"].set(f"{self.live['BROW']:.3f}")
        self.live_vars["cur_x"].set(str(self.cur_x))
        self.live_vars["cur_y"].set(str(self.cur_y))

        for g, (pill, icon, lbl) in self.gest_indicators.items():
            active = self.gesture_active.get(g, False)
            bg = C["green_dim"] if active else C["inactive"]
            fg = C["green"]     if active else C["text3"]
            hi = C["green"]     if active else C["border"]
            pill.config(bg=bg, highlightbackground=hi)
            icon.config(bg=bg, fg=fg if not active else C["green"])
            lbl.config(bg=bg, fg=fg)

    # ── Controls ───────────────────────────────────────────────────────────────
    def _toggle_tracking(self):
        if self.running:
            self.running = self.paused = False
            self.btn_start.config(text="▶   ENGAGE FACE MOUSE",
                                  bg=C["green_dim"], fg=C["green"])
            self._set_status("IDLE", C["text3"], C["inactive"], C["border2"])
        else:
            if self.calibrating:
                messagebox.showwarning("FaceNav",
                    "Calibration not complete.\nHold still and wait.")
                return
            self.running = True
            self.btn_start.config(text="■   DISENGAGE",
                                  bg=C["red_dim"], fg=C["red"])
            self._set_status("TRACKING", C["green"], C["green_dim"], C["green"])

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self._set_status("PAUSED", C["yellow"], C["yellow_dim"], C["yellow"])
        else:
            self._set_status("TRACKING", C["green"], C["green_dim"], C["green"])

    def _recalibrate(self):
        self.calibrating = True
        self.calib_samples = []
        self.calib_fill.config(bg=C["yellow"], width=0)
        self.calib_status.config(
            text="  ◌  RECALIBRATING  —  HOLD STILL",
            fg=C["yellow"])

    def _set_status(self, text, fg, bg, border):
        self.header.itemconfig("status_txt", text=text, fill=fg)
        self.header.itemconfig("status_dot", fill=fg)
        self.header.itemconfig("status_bg", fill=bg, outline=border)

    def _toast(self, msg, color=None):
        color = color or C["green"]
        self.toast_var.set(f"  {msg}  ")
        self.toast_lbl.config(bg=C["green_dim"], fg=color,
                              highlightbackground=color)
        self.toast_lbl.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)
        if self._toast_id:
            self.root.after_cancel(self._toast_id)
        self._toast_id = self.root.after(2500, self.toast_lbl.place_forget)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceNavApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()