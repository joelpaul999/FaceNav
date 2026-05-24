# FaceNav — Biometric Cursor (FaceNav)

**Overview**
- **Purpose:** FaceNav turns facial gestures and head movement into mouse input using MediaPipe face landmarks and OpenCV video feed.
- **UI:** A Tkinter GUI provides calibration, live biometric readouts, and gesture→action mapping.

**Requirements**
- **Python:** 3.8–3.11 recommended; ensure Tcl/Tk is installed for `tkinter` on Windows.
- **Packages:** `mediapipe`, `opencv-contrib-python` (or `opencv-python`), `numpy`, `pyautogui`, `Pillow`.

**Install (Windows)**
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer non-contrib OpenCV, install `opencv-python` instead of `opencv-contrib-python`.

**Model file**
- Place `face_landmarker.task` in the same folder as the scripts before running. The app checks for this file on launch.

**Run**
- From the IDE folder run one of the scripts:

```powershell
python FaceNav6_Edit.py
# or
python FaceNav_6.py
```

**How it works (short)**
- Uses MediaPipe Tasks FaceLandmarker to get facial landmarks each frame.
- Nose tip motion is mapped to screen cursor movement (sensitivity + smoothing).
- Gestures detected: Left Wink, Right Wink, Smile, Raise Eyebrows, Both Eyes Closed — mapped to actions configurable in the GUI.

**Calibration & Controls**
- The app auto-calibrates neutral head position on first launch (keep still until calibration completes).
- Recalibrate from the UI if tracking feels off.
- Gesture mappings are editable via the `GESTURE → ACTION BIND` comboboxes.

**Windows notes & permissions**
- Ensure Python installation includes `tkinter` (Tcl/Tk). If `tkinter` import fails, install a Python distribution that bundles it.
- `pyautogui` controls the mouse — tests may move your cursor; be prepared to stop the script (Alt+F4) if needed.
- Mediapipe binary wheels are platform-specific. Use the recommended Python versions for best compatibility.

**Troubleshooting**
- *Camera not detected:* verify webcam works in other apps and the correct device index (default 0) is available.
- *Model missing:* place `face_landmarker.task` next to the script.
- *Mediapipe install errors:* install a supported Python version or consult MediaPipe install docs for Windows.

**Files**
- Main UI scripts: `FaceNav6_Edit.py`, `FaceNav_6.py`
- Dependencies list: `requirements.txt`
- Model (not included): `face_landmarker.task` — obtain separately.

**Safety & privacy**
- The app processes webcam frames locally and saves screenshots only when the configured gesture action triggers them.
- Use responsibly and be mindful of privacy when running webcams.

---
Created for quick setup and testing. If you want, I can also generate a `setup.bat` for Windows to automate venv creation and install.
