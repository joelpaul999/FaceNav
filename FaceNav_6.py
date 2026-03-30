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
import math

# ══════════════════════════════════════════════════════════════════════════════
# THEME — Cyberpunk HUD
# ══════════════════════════════════════════════════════════════════════════════
SETTINGS_FILE = "sv_settings.json"

C = {
    "bg":         "#030508",
    "bg2":        "#080d14",
    "panel":      "#0a1020",
    "card":       "#0c1424",
    "card_hi":    "#101828",
    "border":     "#0f2040",
    "border2":    "#1a3060",
    "border_hot": "#00aaff",
    "cyan":       "#00e5ff",
    "cyan_dim":   "#00354a",
    "cyan_mid":   "#0077aa",
    "green":      "#00ff9d",
    "green_dim":  "#00261a",
    "red":        "#ff2d55",
    "red_dim":    "#2a0010",
    "yellow":     "#ffe600",
    "yellow_dim": "#1a1600",
    "purple":     "#bf7fff",
    "purple_dim": "#1a0033",
    "orange":     "#ff8800",
    "text":       "#ccdeff",
    "text2":      "#5a7a9a",
    "text3":      "#203050",
    "inactive":   "#0a1428",
}

NOSE_TIP     = 1
CALIB_FRAMES = 40
L_EAR_IDX    = [33,  160, 158, 133, 153, 144]
R_EAR_IDX    = [362, 385, 387, 263, 373, 380]
L_BROW_IDX   = [105, 159]
R_BROW_IDX   = [334, 386]

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

_base    = python.BaseOptions(model_asset_path="face_landmarker.task")
_opts    = vision.FaceLandmarkerOptions(
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
    # Cheek-based smile: mouth width normalised by face width (outer eye corners).
    # More stable than mouth-height ratio — robust to open-mouth / talking.
    mw = np.linalg.norm(
        np.array([lm[61].x, lm[61].y]) - np.array([lm[291].x, lm[291].y]))
    fw = np.linalg.norm(
        np.array([lm[33].x,  lm[33].y])  - np.array([lm[263].x, lm[263].y]))
    return mw / (fw + 1e-6)

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
# WELCOME WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class WelcomeWindow:
    """Splash / pre-flight window shown before the main FaceNav app opens."""

    HOW_TO = [
        ("◈  FACE THE CAMERA",   "Sit centred, ~50 cm from camera. Keep your face well-lit."),
        ("◌  HOLD STILL",        "FaceNav auto-calibrates your neutral position on launch."),
        ("⌣  SMILE",             "Wide smile  →  Double Click  (configurable in main window)"),
        ("◑  LEFT WINK",         "Close left eye only  →  Left Click"),
        ("◐  RIGHT WINK",        "Close right eye only  →  Right Click"),
        ("△  RAISE EYEBROWS",    "Both brows up  →  Scroll Up"),
        ("⊙  BOTH EYES CLOSED",  "Close both eyes  →  Pause / Resume mouse movement"),
    ]

    CHECKS = [
        ("camera", "WEBCAM",               "webcam device"),
        ("model",  "face_landmarker.task", "model file"),
    ]

    def __init__(self, root, on_launch):
        self.root      = root
        self.on_launch = on_launch
        self._check_idx = 0

        self.root.title("FaceNav — Welcome")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)
        self._centre_window(840, 690)

        self._build()
        self._animate_accent()
        self.root.after(300, self._run_checks)   # short delay before checks begin

    def _centre_window(self, w, h):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        # Animated header bar
        self.accent_canvas = tk.Canvas(self.root, bg=C["panel"],
                                       height=66, highlightthickness=0)
        self.accent_canvas.pack(fill="x")
        self.accent_canvas.create_rectangle(0, 0, 2000, 2,
            fill=C["cyan"], outline="", tags="accentline")
        self.accent_canvas.create_text(28, 33, text="◈",
            font=("Segoe UI Symbol", 22), fill=C["cyan"], anchor="w")
        self.accent_canvas.create_text(64, 27, text="FACE",
            font=("Courier New", 18, "bold"), fill=C["text"], anchor="w")
        self.accent_canvas.create_text(130, 27, text="NAV",
            font=("Courier New", 18, "bold"), fill=C["cyan"], anchor="w")
        self.accent_canvas.create_text(64, 47, text="biometric cursor system  v4.0",
            font=("Courier New", 8), fill=C["text3"], anchor="w")
        tk.Frame(self.root, bg=C["border2"], height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=28, pady=16)

        # ── How-to section ────────────────────────────────────────────────────
        self._section_label(body, "HOW TO USE")

        how_outer = tk.Frame(body, bg=C["card"],
                             highlightbackground=C["border2"],
                             highlightthickness=1)
        how_outer.pack(fill="x", pady=(0, 14))
        how_inner = tk.Frame(how_outer, bg=C["card"], padx=12, pady=8)
        how_inner.pack(fill="x")

        for i, (glyph_label, desc) in enumerate(self.HOW_TO):
            row_bg = C["card"] if i % 2 == 0 else C["card_hi"]
            row = tk.Frame(how_inner, bg=row_bg)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=glyph_label,
                     font=("Courier New", 9, "bold"),
                     bg=row_bg, fg=C["cyan"],
                     width=22, anchor="w").pack(side="left", padx=(6, 0), pady=4)
            tk.Label(row, text=desc,
                     font=("Courier New", 9),
                     bg=row_bg, fg=C["text2"],
                     anchor="w").pack(side="left", padx=8)

        # ── Pre-flight checklist ───────────────────────────────────────────────
        self._section_label(body, "PRE-FLIGHT CHECKLIST")

        check_outer = tk.Frame(body, bg=C["card"],
                               highlightbackground=C["border2"],
                               highlightthickness=1)
        check_outer.pack(fill="x", pady=(0, 14))
        check_inner = tk.Frame(check_outer, bg=C["card"], padx=12, pady=12)
        check_inner.pack(fill="x")

        self._check_vars = {}
        for key, label, desc in self.CHECKS:
            row = tk.Frame(check_inner, bg=C["card"])
            row.pack(fill="x", pady=4)

            icon_var   = tk.StringVar(value="○")
            icon_lbl   = tk.Label(row, textvariable=icon_var,
                                  font=("Courier New", 13, "bold"),
                                  bg=C["card"], fg=C["text3"], width=3)
            icon_lbl.pack(side="left")

            tk.Label(row, text=f"  {label}",
                     font=("Courier New", 9, "bold"),
                     bg=C["card"], fg=C["text2"],
                     width=28, anchor="w").pack(side="left")

            status_lbl = tk.Label(row, text=f"waiting to check {desc}…",
                                  font=("Courier New", 9),
                                  bg=C["card"], fg=C["text3"], anchor="w")
            status_lbl.pack(side="left")

            self._check_vars[key] = (icon_var, icon_lbl, status_lbl)

        # ── Initialisation bar ────────────────────────────────────────────────
        self._section_label(body, "SYSTEM INITIALISATION")

        bar_bg = tk.Frame(body, bg=C["border"], height=6)
        bar_bg.pack(fill="x", pady=(0, 5))
        self._init_fill = tk.Frame(bar_bg, bg=C["cyan"], height=6, width=0)
        self._init_fill.place(x=0, y=0, relheight=1)

        self._init_label = tk.Label(body,
            text="  ◌  WAITING…",
            font=("Courier New", 9, "bold"),
            bg=C["bg"], fg=C["text3"], anchor="w")
        self._init_label.pack(fill="x")

        # ── Launch button ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(body, bg=C["bg"])
        btn_frame.pack(fill="x", pady=(16, 0))

        self.launch_btn = tk.Button(btn_frame,
            text="▶   LAUNCH FACENAV",
            font=("Courier New", 11, "bold"),
            bg=C["inactive"], fg=C["text3"],
            activebackground=C["green_dim"],
            activeforeground=C["green"],
            relief="flat", bd=0,
            padx=20, pady=13,
            cursor="arrow",
            command=self._launch,
            state="disabled")
        self.launch_btn.pack(fill="x")

    def _section_label(self, parent, text, top=0):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", pady=(top + 4, 5))
        tk.Label(f, text="▸", font=("Courier New", 9),
                 bg=C["bg"], fg=C["cyan"]).pack(side="left")
        tk.Label(f, text=f" {text}",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text2"]).pack(side="left")
        tk.Frame(f, bg=C["border2"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=5)

    # ── Animated accent ────────────────────────────────────────────────────────
    def _animate_accent(self):
        t   = time.time()
        val = int(120 + 80 * abs(((t * 0.8) % 1.0) - 0.5) * 2)
        val = min(255, max(60, val))
        hex_c = f"#00{val:02x}ff" if val < 230 else "#00e5ff"
        try:
            self.accent_canvas.itemconfig("accentline", fill=hex_c)
        except Exception:
            pass
        self.root.after(60, self._animate_accent)

    # ── Sequential pre-flight checks ──────────────────────────────────────────
    def _run_checks(self):
        self._check_sequence = list(self.CHECKS)
        self._check_idx      = 0
        self._init_label.config(text="  ◌  RUNNING CHECKS…", fg=C["yellow"])
        self._step_check()

    def _step_check(self):
        if self._check_idx >= len(self._check_sequence):
            self._all_checks_done()
            return

        key, label, desc = self._check_sequence[self._check_idx]
        icon_var, icon_lbl, status_lbl = self._check_vars[key]

        icon_var.set("◌")
        icon_lbl.config(fg=C["yellow"])
        status_lbl.config(text=f"  checking {desc}…", fg=C["yellow"])

        # Update bar: partial fill per item
        total   = len(self._check_sequence)
        pct     = int((self._check_idx / total) * 80)   # 0→80 during checks
        self._set_bar(pct, C["cyan"])

        self.root.after(700, lambda: self._resolve(key, label, desc))

    def _resolve(self, key, label, desc):
        icon_var, icon_lbl, status_lbl = self._check_vars[key]
        ok     = False
        detail = ""

        if key == "camera":
            try:
                cap = cv2.VideoCapture(0)
                ok  = cap.isOpened()
                cap.release()
                detail = "  ●  DETECTED" if ok else "  ✗  NOT DETECTED"
            except Exception:
                detail = "  ✗  NOT DETECTED"

        elif key == "model":
            ok     = os.path.exists("face_landmarker.task")
            detail = ("  ●  FOUND" if ok
                      else "  ✗  MISSING — place face_landmarker.task in script folder")

        if ok:
            icon_var.set("✓"); icon_lbl.config(fg=C["green"])
            status_lbl.config(text=detail, fg=C["green"])
        else:
            icon_var.set("✗"); icon_lbl.config(fg=C["red"])
            status_lbl.config(text=detail, fg=C["red"])

        self._check_idx += 1
        self.root.after(350, self._step_check)

    def _all_checks_done(self):
        self._set_bar(100, C["green"])
        self._init_label.config(
            text="  ✓  ALL SYSTEMS READY  —  CLICK LAUNCH TO BEGIN",
            fg=C["green"])
        self._enable_launch()

    def _set_bar(self, pct, color):
        bar_w = self.root.winfo_width() - 56   # window width minus side padding
        if bar_w < 10:
            bar_w = 784
        self._init_fill.config(width=int(bar_w * pct / 100), bg=color)

    def _enable_launch(self):
        self.launch_btn.config(
            state="normal",
            bg=C["green_dim"], fg=C["green"],
            cursor="hand2")
        def on_enter(e):
            self.launch_btn.config(highlightbackground=C["green"],
                                   highlightthickness=1)
        def on_leave(e):
            self.launch_btn.config(highlightthickness=0)
        self.launch_btn.bind("<Enter>", on_enter)
        self.launch_btn.bind("<Leave>", on_leave)

    def _launch(self):
        self.root.destroy()
        self.on_launch()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class FaceNavApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceNav")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)
        self.root.geometry("1320x900")
        self.root.minsize(1100, 780)

        # State
        self.cap     = cv2.VideoCapture(0)
        self.running = False
        self.paused  = False

        self.calibrating         = True
        self.calib_samples       = []
        self.calib_smile_samples = []
        self.smile_baseline      = None
        self.origin_x = self.origin_y = None
        self.smooth_x  = screen_w // 2
        self.smooth_y  = screen_h // 2
        self._calib_pct = 0

        self.sensitivity  = tk.DoubleVar(value=2.5)
        self.smoothing    = tk.DoubleVar(value=0.2)
        self.eye_thresh   = tk.DoubleVar(value=0.20)
        self.smile_thresh = tk.DoubleVar(value=0.08)
        self.brow_thresh  = tk.DoubleVar(value=0.06)
        self.mapping      = {g: tk.StringVar(value=a) for g, a in DEFAULT_MAPPING.items()}
        self.cooldowns    = {g: 0 for g in DEFAULT_MAPPING}

        self.face_detected  = False
        self.cur_x = self.cur_y = 0
        self.live   = {"L EAR": 0.0, "R EAR": 0.0, "SMILE": 0.0, "BROW": 0.0}
        self.gesture_active = {g: False for g in DEFAULT_MAPPING}

        self._toast_id = None
        self._cam_w    = 600   # camera display width

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
            self.smile_thresh.set(s.get("smile_thresh",0.08))
            self.brow_thresh.set(s.get("brow_thresh",  0.06))
            for g, a in s.get("mapping", {}).items():
                if g in self.mapping: self.mapping[g].set(a)
        except Exception: pass

    def _save_settings(self):
        s = {
            "sensitivity":  self.sensitivity.get(),
            "smoothing":    self.smoothing.get(),
            "eye_thresh":   self.eye_thresh.get(),
            "smile_thresh": self.smile_thresh.get(),
            "brow_thresh":  self.brow_thresh.get(),
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
    # BUILD UI  —  Two-column layout
    #   LEFT  : camera (600×450) + gesture activity + biometric readout
    #   RIGHT : gesture→action mapper  +  system controls  +  sliders (2-col)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── HEADER ────────────────────────────────────────────────────────────
        self.header = tk.Canvas(self.root, bg=C["panel"],
                                height=58, highlightthickness=0)
        self.header.pack(fill="x")
        self.header.create_rectangle(0, 0, 2000, 2,
            fill=C["cyan"], outline="", tags="accentline")
        self.header.create_text(24, 29, text="◈",
            font=("Segoe UI Symbol", 20), fill=C["cyan"], anchor="w")
        self.header.create_text(54, 24, text="FACE",
            font=("Courier New", 16, "bold"), fill=C["text"], anchor="w")
        self.header.create_text(110, 24, text="NAV",
            font=("Courier New", 16, "bold"), fill=C["cyan"], anchor="w")
        self.header.create_text(54, 42, text="biometric cursor system  v4.0",
            font=("Courier New", 8), fill=C["text3"], anchor="w")

        self.header.create_rectangle(0, 0, 0, 0,
            fill=C["inactive"], outline=C["border2"], width=1, tags="status_bg")
        self.header.create_text(0, 0, text="⬤",
            font=("Courier New", 9), fill=C["text3"], tags="status_dot")
        self.header.create_text(0, 0, text="IDLE",
            font=("Courier New", 11, "bold"), fill=C["text3"], tags="status_txt")
        self.header.bind("<Configure>", self._reposition_status)

        tk.Frame(self.root, bg=C["border2"], height=1).pack(fill="x")

        # ── BODY ──────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT COLUMN
        # ═════════════════════════════════════════════════════════════════════
        left = tk.Frame(body, bg=C["bg"])
        left.pack(side="left", fill="y", padx=16, pady=16)

        # Camera card — 600×450
        cam_card = tk.Frame(left, bg=C["border_hot"], padx=1, pady=1)
        cam_card.pack()
        cam_body = tk.Frame(cam_card, bg=C["panel"])
        cam_body.pack()

        cam_strip = tk.Frame(cam_body, bg=C["card"], height=30)
        cam_strip.pack(fill="x")
        cam_strip.pack_propagate(False)
        tk.Label(cam_strip, text="  SYS::CAM-0   INPUT FEED",
                 font=("Courier New", 8, "bold"),
                 bg=C["card"], fg=C["text2"]).pack(side="left", pady=6)
        self.face_badge = tk.Label(cam_strip, text=" NO FACE DETECTED ",
                                   font=("Courier New", 7, "bold"),
                                   bg=C["red_dim"], fg=C["red"], padx=6)
        self.face_badge.pack(side="right", padx=8, pady=6)

        self.cam_label = tk.Label(cam_body, bg="#000000",
                                  width=self._cam_w, height=450)
        self.cam_label.pack()

        # Calibration progress bar
        prog_bg = tk.Frame(cam_body, bg=C["border"], height=5)
        prog_bg.pack(fill="x")
        self.calib_fill = tk.Frame(prog_bg, bg=C["yellow"], height=5, width=0)
        self.calib_fill.place(x=0, y=0, relheight=1)

        self.calib_status = tk.Label(cam_body,
            text="  ◌  CALIBRATING  —  KEEP NEUTRAL",
            font=("Courier New", 9, "bold"),
            bg=C["panel"], fg=C["yellow"],
            anchor="w", pady=6)
        self.calib_status.pack(fill="x")

        # Gesture activity pills
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
                            highlightthickness=1, padx=10, pady=6)
            pill.pack(side="left", padx=4)
            g_icon = tk.Label(pill, text=GESTURE_GLYPHS[g],
                              font=("Courier New", 14),
                              bg=C["inactive"], fg=C["text3"])
            g_icon.pack(side="left")
            g_lbl = tk.Label(pill, text=f" {short[g]}",
                             font=("Courier New", 7, "bold"),
                             bg=C["inactive"], fg=C["text3"])
            g_lbl.pack(side="left")
            self.gest_indicators[g] = (pill, g_icon, g_lbl)

        # Biometric readout — 3-column grid
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
            ("SMILE",  "smile", C["purple"]),
            ("BROW",   "brow",  C["yellow"]),
            ("CUR-X",  "cur_x", C["green"]),
            ("CUR-Y",  "cur_y", C["green"]),
        ]
        for i, (label, key, color) in enumerate(cells):
            col   = i % 3
            row_i = i // 3
            cell  = tk.Frame(inner_grid, bg=C["bg2"],
                             highlightbackground=C["border"],
                             highlightthickness=1, padx=10, pady=7)
            cell.grid(row=row_i, column=col, padx=5, pady=3, sticky="ew")
            tk.Label(cell, text=label, font=("Courier New", 7, "bold"),
                     bg=C["bg2"], fg=C["text3"]).pack(anchor="w")
            v = tk.StringVar(value="—")
            self.live_vars[key] = v
            tk.Label(cell, textvariable=v, font=("Courier New", 15, "bold"),
                     bg=C["bg2"], fg=color, width=8, anchor="w").pack()

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT COLUMN — mapper + controls + sliders
        # ═════════════════════════════════════════════════════════════════════
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True,
                   padx=(0, 16), pady=16)

        # Gesture → Action mapper
        self._hud_label(right, "GESTURE → ACTION BIND")

        map_outer = tk.Frame(right, bg=C["card"],
                             highlightbackground=C["border2"],
                             highlightthickness=1)
        map_outer.pack(fill="x", pady=(0, 4))

        for i, (gesture, var) in enumerate(self.mapping.items()):
            row_bg = C["card"] if i % 2 == 0 else C["card_hi"]
            row = tk.Frame(map_outer, bg=row_bg)
            row.pack(fill="x")
            ri = tk.Frame(row, bg=row_bg, pady=8, padx=12)
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
                         state="readonly", width=18,
                         style="FN.TCombobox").pack(side="right")

        # System controls — horizontal row of 3 buttons
        self._hud_label(right, "SYSTEM CONTROLS", top=14)

        ctrl_row = tk.Frame(right, bg=C["bg"])
        ctrl_row.pack(fill="x", pady=(0, 8))
        ctrl_row.columnconfigure(0, weight=1)
        ctrl_row.columnconfigure(1, weight=1)
        ctrl_row.columnconfigure(2, weight=1)

        self.btn_start = self._hud_btn_grid(ctrl_row,
            "▶   ENGAGE FACE MOUSE",
            self._toggle_tracking, C["green"], C["green_dim"], col=0)
        self._hud_btn_grid(ctrl_row, "⟳   RECALIBRATE",
            self._recalibrate, C["cyan"], C["cyan_dim"], col=1)
        self._hud_btn_grid(ctrl_row, "↓   COMMIT SETTINGS",
            self._save_settings, C["purple"], C["purple_dim"], col=2)

        # Detection parameter sliders — 2-column grid
        self._hud_label(right, "DETECTION PARAMETERS", top=6)

        slider_grid = tk.Frame(right, bg=C["bg"])
        slider_grid.pack(fill="both", expand=True)
        slider_grid.columnconfigure(0, weight=1)
        slider_grid.columnconfigure(1, weight=1)

        # zone_fn(value) → (zone_label, zone_color) for each slider
        # Zones give a plain-English reading of where the knob currently sits.
        def _eye_zone(v):
            if v < 0.13: return ("TOO SENSITIVE — may false-trigger on blinks", C["red"])
            if v < 0.18: return ("SENSITIVE — good for slow/partial blinks",    C["yellow"])
            if v < 0.26: return ("BALANCED — recommended for most users",        C["green"])
            if v < 0.33: return ("FIRM — requires a deliberate full blink",      C["yellow"])
            return              ("VERY FIRM — hard to trigger intentionally",     C["red"])

        def _smile_zone(v):
            if v < 0.04: return ("TOO EASY — neutral expression may fire",       C["red"])
            if v < 0.07: return ("EASY — slight smile triggers it",              C["yellow"])
            if v < 0.12: return ("BALANCED — wide smile required",               C["green"])
            if v < 0.17: return ("FIRM — needs an exaggerated smile",            C["yellow"])
            return              ("VERY FIRM — smile rarely detected",             C["red"])

        def _brow_zone(v):
            if v < 0.04: return ("TOO SENSITIVE — micro-movements may fire",     C["red"])
            if v < 0.07: return ("SENSITIVE — subtle raise triggers it",         C["yellow"])
            if v < 0.10: return ("BALANCED — clear raise required",              C["green"])
            if v < 0.13: return ("FIRM — needs an obvious raise",                C["yellow"])
            return              ("VERY FIRM — hard to trigger reliably",          C["red"])

        def _sens_zone(v):
            if v < 2.0:  return ("SLOW — large head movements needed",           C["yellow"])
            if v < 3.5:  return ("COMFORTABLE — good for general use",           C["green"])
            if v < 5.5:  return ("FAST — small movements cover full screen",     C["yellow"])
            return              ("VERY FAST — may be hard to control precisely",  C["red"])

        def _smooth_zone(v):
            if v < 0.10: return ("VERY SMOOTH — noticeable input lag",           C["yellow"])
            if v < 0.20: return ("SMOOTH — fluid, slight lag",                   C["green"])
            if v < 0.35: return ("RESPONSIVE — minimal lag, slight jitter",      C["green"])
            return              ("RAW — very responsive but jittery",             C["yellow"])

        sliders = [
            ("EYE BLINK THRESHOLD",  self.eye_thresh,   0.05, 0.40, C["cyan"],
             "EAR value below which eye is considered closed", _eye_zone),
            ("SMILE SENSITIVITY",    self.smile_thresh,  0.02, 0.20, C["purple"],
             "Margin above calibrated neutral — raise if triggering without smiling", _smile_zone),
            ("BROW RAISE THRESHOLD", self.brow_thresh,   0.02, 0.15, C["yellow"],
             "Vertical brow-to-eye distance for eyebrow raise", _brow_zone),
            ("CURSOR SENSITIVITY",   self.sensitivity,   1.0,  8.0,  C["green"],
             "Amplification of nose movement to screen movement", _sens_zone),
            ("MOVEMENT SMOOTHING",   self.smoothing,     0.05, 0.5,  C["orange"],
             "Lerp factor — lower = smoother, higher = snappier", _smooth_zone),
        ]

        for idx, (label, var, lo, hi, color, hint, zone_fn) in enumerate(sliders):
            col   = idx % 2
            row_i = idx // 2

            card = tk.Frame(slider_grid, bg=C["card"],
                            highlightbackground=C["border2"],
                            highlightthickness=1)
            card.grid(row=row_i, column=col,
                      padx=(0, 6) if col == 0 else (0, 0),
                      pady=5, sticky="nsew")
            slider_grid.rowconfigure(row_i, weight=1)

            ci = tk.Frame(card, bg=C["card"], padx=14, pady=11)
            ci.pack(fill="both", expand=True)

            hrow = tk.Frame(ci, bg=C["card"])
            hrow.pack(fill="x")
            tk.Frame(hrow, bg=color, width=3, height=16).pack(side="left")
            tk.Label(hrow, text=f"  {label}",
                     font=("Segoe UI", 12, "bold"),
                     bg=C["card"], fg=C["text"]).pack(side="left")
            val_var = tk.StringVar()
            tk.Label(hrow, textvariable=val_var,
                     font=("Consolas", 18, "bold"),
                     bg=C["card"], fg="white",
                     width=6, anchor="e").pack(side="right")

            tk.Label(ci, text=hint,
                     font=("Segoe UI", 10),
                     bg=C["card"], fg=C["text2"],
                     anchor="w", wraplength=250).pack(fill="x", pady=(4, 6))

            srow = tk.Frame(ci, bg=C["card"])
            srow.pack(fill="x")
            tk.Label(srow, text=f"{lo:.2f}",
                     font=("Consolas", 10),
                     bg=C["card"], fg=C["text2"]).pack(side="left")
            sl = tk.Scale(srow, variable=var,
                          from_=lo, to=hi, resolution=0.01,
                          orient="horizontal",
                          bg=C["card"], fg=color,
                          troughcolor=C["bg2"],
                          activebackground=color,
                          highlightthickness=0, bd=0,
                          showvalue=False)
            sl.pack(side="left", fill="x", expand=True, padx=(4, 4))
            tk.Label(srow, text=f"{hi:.2f}",
                     font=("Consolas", 10),
                     bg=C["card"], fg=C["text2"]).pack(side="left")

            # Zone indicator — live coloured status line below the slider
            zone_frame = tk.Frame(ci, bg=C["bg2"],
                                  highlightbackground=C["border"],
                                  highlightthickness=1)
            zone_frame.pack(fill="x", pady=(6, 0))
            zone_dot = tk.Label(zone_frame, text=" ⬤ ",
                                font=("Courier New", 9),
                                bg=C["bg2"], fg=C["text3"])
            zone_dot.pack(side="left", padx=(4, 0), pady=3)
            zone_lbl = tk.Label(zone_frame, text="",
                                font=("Courier New", 9, "bold"),
                                bg=C["bg2"], fg=C["text3"],
                                anchor="w")
            zone_lbl.pack(side="left", pady=3)

            def _trace(v=var, vv=val_var, zfn=zone_fn, zd=zone_dot, zl=zone_lbl):
                def cb(*_):
                    val = v.get()
                    vv.set(f"{val:.2f}")
                    ztxt, zcol = zfn(val)
                    zd.config(fg=zcol)
                    zl.config(text=ztxt, fg=zcol)
                v.trace_add("write", cb); cb()
            _trace()

        # ── Sixth cell (row 2, col 1) — live tuning advisor ───────────────────
        advisor_card = tk.Frame(slider_grid, bg=C["card"],
                                highlightbackground=C["border2"],
                                highlightthickness=1)
        advisor_card.grid(row=2, column=1,
                          padx=(0, 0), pady=5, sticky="nsew")

        ai = tk.Frame(advisor_card, bg=C["card"], padx=14, pady=11)
        ai.pack(fill="both", expand=True)

        # Header
        ah = tk.Frame(ai, bg=C["card"])
        ah.pack(fill="x")
        tk.Frame(ah, bg=C["cyan"], width=3, height=16).pack(side="left")
        tk.Label(ah, text="  TUNING ADVISOR",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["card"], fg=C["text"]).pack(side="left")
        self._advisor_dot = tk.Label(ah, text="⬤",
                                     font=("Courier New", 12),
                                     bg=C["card"], fg=C["green"])
        self._advisor_dot.pack(side="right")

        tk.Label(ai, text="Overall profile based on current settings",
                 font=("Segoe UI", 10),
                 bg=C["card"], fg=C["text2"],
                 anchor="w").pack(fill="x", pady=(4, 8))

        # Five mini rows — one per parameter
        advisor_params = [
            ("EYE",   self.eye_thresh,   _eye_zone,   C["cyan"]),
            ("SMILE", self.smile_thresh, _smile_zone, C["purple"]),
            ("BROW",  self.brow_thresh,  _brow_zone,  C["yellow"]),
            ("SPEED", self.sensitivity,  _sens_zone,  C["green"]),
            ("SMOOTH",self.smoothing,    _smooth_zone,C["orange"]),
        ]
        self._advisor_rows = []
        for param_label, param_var, param_zfn, param_color in advisor_params:
            prow = tk.Frame(ai, bg=C["bg2"],
                            highlightbackground=C["border"],
                            highlightthickness=1)
            prow.pack(fill="x", pady=2)

            tk.Label(prow, text=f" {param_label}",
                     font=("Courier New", 9, "bold"),
                     bg=C["bg2"], fg=param_color,
                     width=7, anchor="w").pack(side="left", padx=(2, 0), pady=3)

            # Mini fill bar (canvas)
            bar_canvas = tk.Canvas(prow, bg=C["bg2"],
                                   height=10, highlightthickness=0,
                                   width=100)
            bar_canvas.pack(side="left", padx=4, pady=4)

            status_lbl = tk.Label(prow, text="",
                                  font=("Courier New", 8),
                                  bg=C["bg2"], fg=C["text3"],
                                  anchor="w")
            status_lbl.pack(side="left", padx=(2, 4), pady=3)

            self._advisor_rows.append(
                (param_var, param_zfn, param_color, bar_canvas, status_lbl))

        # Wire up live updates for the advisor
        def _update_advisor(*_):
            all_green = True
            any_red   = False
            for (pvar, pzfn, pcol, bcanv, slbl) in self._advisor_rows:
                val = pvar.get()
                ztxt, zcol = pzfn(val)
                # Shorten the zone text to the first word or two for the mini row
                short = ztxt.split("—")[0].strip()
                slbl.config(text=short, fg=zcol)
                # Draw mini bar: fill proportion = (val - lo) / (hi - lo)
                # We reuse lo/hi from the slider defs via pzfn signature
                bcanv.delete("all")
                bcanv.create_rectangle(0, 2, 100, 8,
                                       fill=C["border"], outline="")
                fill_w = min(100, max(4, int(100 * 0.5)))  # midpoint placeholder
                bcanv.create_rectangle(0, 2, fill_w, 8,
                                       fill=zcol, outline="")
                if zcol == C["red"]:   any_red   = True
                if zcol != C["green"]: all_green = False

            # Overall dot
            if any_red:
                self._advisor_dot.config(fg=C["red"])
            elif all_green:
                self._advisor_dot.config(fg=C["green"])
            else:
                self._advisor_dot.config(fg=C["yellow"])

        for pvar, *_ in self._advisor_rows:
            pvar.trace_add("write", _update_advisor)
        _update_advisor()

        # Toast
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
        tk.Label(f, text="▸", font=("Courier New", 9),
                 bg=C["bg"], fg=C["cyan"]).pack(side="left")
        tk.Label(f, text=f" {text}",
                 font=("Segoe UI", 9, "bold"),
                 bg=C["bg"], fg=C["text2"]).pack(side="left")
        tk.Frame(f, bg=C["border2"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=5)

    def _hud_btn_grid(self, parent, text, cmd, fg, bg, col):
        """Button placed in a grid column (for horizontal control row)."""
        btn = tk.Button(parent, text=text, command=cmd,
                        font=("Courier New", 8, "bold"),
                        bg=bg, fg=fg,
                        activebackground=C["card_hi"],
                        activeforeground=fg,
                        relief="flat", bd=0,
                        padx=10, pady=10,
                        cursor="hand2", anchor="w")
        btn.grid(row=0, column=col, sticky="ew", padx=(0, 6) if col < 2 else 0, pady=2)

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
        self.header.coords("status_bg",  x1, y1, x2, y2)
        cy = (y1 + y2) // 2
        self.header.coords("status_dot", x1 + 18, cy)
        self.header.coords("status_txt", x1 + 34, cy)

    def _animate_header(self):
        t   = time.time()
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

                if self.calibrating:
                    self.calib_samples.append((nx, ny))
                    self.calib_smile_samples.append(smile_ratio(lm))
                    pct = int(len(self.calib_samples) / CALIB_FRAMES * 100)
                    self._calib_pct = pct
                    self.calib_fill.config(width=int(self._cam_w * pct / 100))
                    self.calib_status.config(
                        text=f"  ◌  CALIBRATING  {pct}%  —  KEEP NEUTRAL",
                        fg=C["yellow"])
                    cv2.putText(frame, f"CALIBRATING  {pct}%  — KEEP NEUTRAL",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 229, 255), 2)
                    if len(self.calib_samples) >= CALIB_FRAMES:
                        self.origin_x = np.mean([s[0] for s in self.calib_samples])
                        self.origin_y = np.mean([s[1] for s in self.calib_samples])
                        sm = np.array(self.calib_smile_samples)
                        self.smile_baseline = float(np.mean(sm) + 1.5 * np.std(sm))
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

                    baseline     = self.smile_baseline if self.smile_baseline is not None else 1.0
                    smile_active = smile_v > (baseline + st)

                    gestures = {
                        "Left Wink":        l_closed and not both_shut,
                        "Right Wink":       r_closed and not both_shut,
                        "Smile":            smile_active,
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

                    if self.origin_x:
                        ox, oy = int(self.origin_x), int(self.origin_y)
                        cv2.line(frame, (ox-10, oy), (ox+10, oy), (255, 130, 0), 1)
                        cv2.line(frame, (ox, oy-10), (ox, oy+10), (255, 130, 0), 1)
                        cv2.circle(frame, (int(nx), int(ny)), 7, (0, 229, 255), -1)
                        cv2.line(frame, (ox, oy), (int(nx), int(ny)), (0, 255, 157), 1)

                    for val, lbl, thresh, bx in [
                            (l_ear_v, "L", et, 8),
                            (r_ear_v, "R", et, 84)]:
                        bar = int(np.clip(val * 210, 0, fw - bx - 4))
                        clr = (80, 40, 255) if val < thresh else (0, 255, 157)
                        cv2.rectangle(frame, (bx, fh-20), (bx+bar, fh-10), clr, -1)
                        cv2.putText(frame, lbl, (bx, fh-24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1)

                    margin, length, thick = 8, 18, 2
                    clr_hud = (0, 229, 255)
                    for (x, y, ddx, ddy) in [
                            (margin, margin, 1, 1),
                            (fw-margin, margin, -1, 1),
                            (margin, fh-margin, 1, -1),
                            (fw-margin, fh-margin, -1, -1)]:
                        cv2.line(frame, (x, y), (x+ddx*length, y), clr_hud, thick)
                        cv2.line(frame, (x, y), (x, y+ddy*length), clr_hud, thick)

                    if self.paused and self.running:
                        ov = frame.copy()
                        cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 0), -1)
                        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
                        cv2.putText(frame, "// PAUSED //",
                                    (fw//2 - 95, fh//2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 229, 255), 2)

            # Render at 600×450
            display = cv2.resize(frame, (self._cam_w, 450))
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
        self.calibrating         = True
        self.calib_samples       = []
        self.calib_smile_samples = []
        self.smile_baseline      = None
        self.calib_fill.config(bg=C["yellow"], width=0)
        self.calib_status.config(
            text="  ◌  RECALIBRATING  —  KEEP NEUTRAL",
            fg=C["yellow"])

    def _set_status(self, text, fg, bg, border):
        self.header.itemconfig("status_txt", text=text, fill=fg)
        self.header.itemconfig("status_dot", fill=fg)
        self.header.itemconfig("status_bg",  fill=bg, outline=border)

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
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def launch_main():
    main_root = tk.Tk()
    app = FaceNavApp(main_root)
    main_root.protocol("WM_DELETE_WINDOW", app.on_close)
    main_root.mainloop()


if __name__ == "__main__":
    welcome_root = tk.Tk()
    WelcomeWindow(welcome_root, on_launch=launch_main)
    welcome_root.mainloop()