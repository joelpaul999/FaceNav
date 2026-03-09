import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ── Landmark indices ───────────────────────────────────────────────────────────
NOSE_TIP = 4

# EAR uses 6 points per eye: 2 horizontal corners + 4 vertical points
# Left eye landmarks
L_EAR_IDX = [33, 160, 158, 133, 153, 144]
# Right eye landmarks
R_EAR_IDX = [362, 385, 387, 263, 373, 380]

# ── Smoothing ──────────────────────────────────────────────────────────────────
SMOOTH = 0.2
smooth_x, smooth_y = screen_w // 2, screen_h // 2

# ── Blink state ────────────────────────────────────────────────────────────────
# EAR below this = eye closed. Open eye is ~0.25-0.30, closed is ~0.15 or below
# Tune this if clicks are too sensitive or not sensitive enough
EAR_THRESHOLD = 0.20
left_cooldown  = 0
right_cooldown = 0

# ── Calibration ────────────────────────────────────────────────────────────────
calibrating   = True
calib_samples = []
CALIB_FRAMES  = 30

# ── EAR calculation ────────────────────────────────────────────────────────────
def ear(landmarks, idx):
    """Eye Aspect Ratio — ratio of eye height to width using 6 landmarks."""
    p = [(landmarks[i].x, landmarks[i].y) for i in idx]
    # Vertical distances
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    # Horizontal distance
    h  = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h)

timestamp = 0
print("Hold your head still for calibration...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results  = detector.detect_for_video(mp_image, timestamp)
    timestamp += 33

    if results.face_landmarks:
        face = results.face_landmarks[0]
        nose = face[NOSE_TIP]
        nx, ny = nose.x * w, nose.y * h

        # ── Calibration phase ──────────────────────────────────────────────────
        if calibrating:
            calib_samples.append((nx, ny))
            remaining = CALIB_FRAMES - len(calib_samples)
            cv2.putText(frame, f"Calibrating... look straight ahead ({remaining})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2)
            if len(calib_samples) >= CALIB_FRAMES:
                origin_x = np.mean([s[0] for s in calib_samples])
                origin_y = np.mean([s[1] for s in calib_samples])
                calibrating = False
                print(f"Calibration done. Origin: ({origin_x:.1f}, {origin_y:.1f})")

        # ── Tracking phase ─────────────────────────────────────────────────────
        else:
            dx = nx - origin_x
            dy = ny - origin_y

            SENSITIVITY_X = screen_w / (w * 0.25)
            SENSITIVITY_Y = screen_h / (h * 0.25)

            target_x = np.clip(screen_w / 2 + dx * SENSITIVITY_X, 0, screen_w - 1)
            target_y = np.clip(screen_h / 2 + dy * SENSITIVITY_Y, 0, screen_h - 1)

            smooth_x += SMOOTH * (target_x - smooth_x)
            smooth_y += SMOOTH * (target_y - smooth_y)
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── EAR blink detection ────────────────────────────────────────────
            l_ear = ear(face, L_EAR_IDX)
            r_ear = ear(face, R_EAR_IDX)

            l_closed = l_ear < EAR_THRESHOLD
            r_closed = r_ear < EAR_THRESHOLD
            both_closed = l_closed and r_closed

            # Left eye wink → left click
            if l_closed and not both_closed and left_cooldown == 0:
                pyautogui.click(button="left")
                left_cooldown = 20
            if left_cooldown > 0:
                left_cooldown -= 1

            # Right eye wink → right click
            if r_closed and not both_closed and right_cooldown == 0:
                pyautogui.click(button="right")
                right_cooldown = 20
            if right_cooldown > 0:
                right_cooldown -= 1

            # ── Debug overlay ──────────────────────────────────────────────────
            cv2.circle(frame, (int(nx), int(ny)), 6, (0, 229, 255), -1)
            cv2.circle(frame, (int(origin_x), int(origin_y)), 6, (255, 100, 0), 2)
            cv2.line(frame, (int(origin_x), int(origin_y)),
                     (int(nx), int(ny)), (0, 255, 136), 1)

            cv2.putText(frame, f"dx:{dx:+.1f} dy:{dy:+.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)
            cv2.putText(frame, f"cursor: ({int(smooth_x)}, {int(smooth_y)})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)

            l_label = "IGNORED" if both_closed else ("LEFT CLICK!" if l_closed else "")
            r_label = "IGNORED" if both_closed else ("RIGHT CLICK!" if r_closed else "")
            l_color = (0, 80, 255) if l_closed else (180, 180, 180)
            r_color = (0, 80, 255) if r_closed else (180, 180, 180)

            cv2.putText(frame, f"L EAR: {l_ear:.3f}  {l_label}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
            cv2.putText(frame, f"R EAR: {r_ear:.3f}  {r_label}",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 2)
            cv2.putText(frame, f"threshold: {EAR_THRESHOLD}  [+] raise  [-] lower",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow("Face Mouse — ESC to quit", frame)
    key = cv2.waitKey(1)
    if key == 27:    # ESC → quit
        break
    elif key == ord('+') or key == ord('='):
        EAR_THRESHOLD = round(min(EAR_THRESHOLD + 0.01, 0.40), 3)
        print(f"Threshold: {EAR_THRESHOLD}")
    elif key == ord('-'):
        EAR_THRESHOLD = round(max(EAR_THRESHOLD - 0.01, 0.05), 3)
        print(f"Threshold: {EAR_THRESHOLD}")

cap.release()
cv2.destroyAllWindows()
