# 👁️ Nitriva — Assistive Eye-Tracking & Communication System

> A computer-vision assistive technology prototype that turns gaze, blinks, and winks into accessible interaction commands.

## Overview

**Nitriva** explores how computer vision can support people with severe speech or motor impairments by providing an alternative interaction method based on eye movement and facial gestures.

The system combines **MediaPipe Face Landmarker, iris/eye features, gaze calibration, blink/wink detection, smoothing, and an accessible communication interface**.

## Problem

People with limited motor control may have difficulty interacting with conventional keyboards, mice, or touch interfaces. Nitriva investigates a webcam-based interface where eye gaze and intentional eye gestures can become input signals.

## System Flow

```text
Webcam
  │
  ▼
Face & Landmark Detection
  │
  ▼
Eye / Iris Feature Extraction
  │
  ▼
5-Point Gaze Calibration
  │
  ▼
Gaze Coordinate Estimation
  │
  ▼
EMA Smoothing
  │
  ├──────────────► Gaze Cursor / Focus
  ├──────────────► Blink Navigation
  └──────────────► Wink Selection
                     │
                     ▼
          Communication / Device / SOS
```

## Core Features

- **5-point gaze calibration** for mapping eye features to screen coordinates.
- **478-point face landmark tracking** using MediaPipe.
- Iris-based gaze estimation using eye-region geometry.
- Blendshape signals for eye movement and blink detection.
- **Blink and wink gesture controls** for hands-free navigation.
- Exponential Moving Average smoothing to reduce gaze jitter.
- Communication board for common needs such as water, pain, food, family, and doctor assistance.
- Browser speech output for generated communication sentences.
- Accessibility simulation mode for development and testing without relying entirely on camera input.
- Emergency/SOS interaction concept for urgent assistance.

## Computer Vision Pipeline

1. Capture webcam frames.
2. Detect the face and facial landmarks.
3. Extract iris and eye-region features.
4. Combine geometric and blendshape signals.
5. Calibrate the mapping between eye features and screen coordinates.
6. Smooth noisy coordinates using an exponential moving average.
7. Interpret gaze and intentional gestures as UI commands.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| Computer Vision | MediaPipe |
| Face Tracking | MediaPipe Face Landmarker |
| ML / Signal Processing | Feature fusion + EMA smoothing |
| Interface | Web UI / JavaScript |
| Speech | Browser Web Speech API |
| Development | VS Code |

## Example Interaction Model

| Input | Action |
|---|---|
| Gaze | Move focus / cursor |
| Left wink | Select focused item |
| Right wink | Clear selection |
| Double blink | Navigate back |
| Triple blink | Emergency / logout action |

Keyboard simulation is also available during development so the system can be tested without relying on eye movement alone.

## Project Structure

```text
Nitriva/
├── run.py                         # Application entry point
├── app / server modules           # Application and API logic
├── vision / optics modules        # Face, eye and gaze processing
├── templates/                     # UI templates
├── static/                        # Frontend assets
└── README.md
```

> The exact module names may evolve as the project develops; the architecture above describes the major responsibilities of the system.

## Deploy on Render

Use the Docker Blueprint in `render.yaml` to deploy the frontend and Python backend together. See [DEPLOYMENT.md](DEPLOYMENT.md) for setup, verification, and free-plan limitations.

## Running Locally

```bash
git clone https://github.com/juneja711/Nitriva.git
cd Nitriva
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Then open the local application shown by the server.

## Engineering Challenges

### Gaze calibration

Raw iris coordinates vary between users and camera positions. A calibration stage is therefore used to learn a mapping from eye features to screen coordinates.

### Gaze stability

Small involuntary eye movements and camera noise can make a cursor unstable. Nitriva applies exponential moving average smoothing to reduce jitter.

### Intent detection

A gaze direction alone should not always trigger an action. Blink/wink gestures are used as deliberate commands so navigation can be separated from selection.

## Future Improvements

- Per-user calibration persistence.
- Improved low-light robustness.
- More robust temporal gesture classification.
- On-device inference optimization.
- Accessibility-focused usability testing.
- Hardware/device integration with authenticated control APIs.

---

Built as a hands-on exploration of **computer vision, human-computer interaction, accessibility engineering, and assistive AI systems**.
