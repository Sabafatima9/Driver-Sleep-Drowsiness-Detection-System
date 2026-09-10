<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2563EB,100:14B8A6&height=220&section=header&text=Driver%20Sleep%20and%20Drowsiness%20Detector&fontSize=30&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Real-Time%20Driver%20Monitoring%20System%20(DMS)&descAlignY=55&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2500&pause=500&color=2563EB&center=true&vCenter=true&width=700&lines=%F0%9F%98%B4+PERCLOS+and+Fatigue+Scoring;%F0%9F%9A%A8+Audio-Visual+Safety+Alarms;%F0%9F%A5%B1+Yawn+and+Head+Nod+Detection;%F0%9F%93%9D+Incident+Logging+and+CSV+Export" alt="Typing SVG" />

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/CV-MediaPipe-00C9A7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-FF4B4B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-14B8A6?style=for-the-badge)

</div>

---

## 🎯 Overview

**Driver Sleep & Drowsiness Detection System (DMS)** is a real-time Computer Vision safety app built with **MediaPipe**, **OpenCV**, and **Tkinter**. It performs multi-factor fatigue analysis — eye closure, yawning, and head slump/nodding — and computes a scientific PERCLOS-based danger score, triggering flashing alerts and audio alarms the moment drowsiness is detected, on both live webcam feeds and recorded driving footage.

---

## 🎬 Demo Video

<div align="center">

lace the link above with your uploaded demo video / GIF showing the fatigue score, drowsiness alert banner, and buzzer alarm triggering.)*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 😴 Multi-Factor Fatigue Analysis
- Eye Aspect Ratio (EAR) tracking
- Prolonged micro-sleep eye closure detection
- Mouth Aspect Ratio (MAR) yawning detection
- 3D head nodding / slump detection

</td>
<td width="50%" valign="top">

### 🚨 Safety Alerts & Scoring
- **PERCLOS** — scientific eye-closure percentage metric
- Composite 0–100% fatigue Danger Score
- Green / Yellow / Red status alerts
- Flashing emergency red alert banner
- Asynchronous audio buzzer alarm with mute toggle

</td>
</tr>
</table>

### 🎥 Dual Stream Compatibility
- Live webcam feed monitoring
- Recorded driving video file analysis
- Timeline scrubber and playback speed controls

### 📝 Incident Logging & Export
- Timestamped sleep and yawn events recorded
- One-click CSV export of full incident log

### 🎨 Shared UI Features
- **Dual MediaPipe Architecture** — works with modern MediaPipe Tasks (`>=1.0.0`) and legacy Solutions (`<1.0.0`)
- **3 Display Modes** — Camera + Skeleton, Skeleton Only (Dark Void), Raw Camera Stream
- **Color Themes** — Cyberpunk Neon, Matrix Emerald, Electric Sunset, Sci-Fi Cyan
- **Live Badges** — FPS counter, detected face status, camera/video status
- **Action Controls** — ⏸️ Pause/Resume, 📸 Snapshot to `snapshots/`, 🔄 Switch camera index, 🔇 Mute alarm

---

## 🛠️ Tech Stack

<div align="center">
<img src="https://skillicons.dev/icons?i=python,opencv" />
</div>

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Landmark Detection | Google MediaPipe Face Mesh |
| Video & Rendering | OpenCV |
| Desktop UI | Tkinter (Dark Theme) |
| Audio Alerts | Python audio playback (async buzzer thread) |
| Data Export | CSV (built-in `csv` module) |

---

## 📦 Requirements & Installation

**1. Activate your virtual environment**
```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*(or `pip install -r requirement.txt`)*

---

## ▶️ Running the Application

```bash
python sleep_detector.py
```

> Works on live webcam by default — switch to a recorded driving video via the sidebar file browser, using the timeline scrubber and speed controls for playback review.

---

## 🏗️ Project Structure

```
Driver_Sleep_Detector/
├── sleep_detector.py         # 🎮 Entry point — Tkinter UI + fatigue analysis pipeline
├── assets/
│   └── alarm.wav                # Buzzer alert sound
├── snapshots/                     # 📸 Saved snapshots
├── logs/                            # 📝 Exported CSV incident logs
├── requirements.txt                   # Dependencies
└── README.md                         # 📖 You're here
```

---

## 🧠 How It Works

| Concept | Implementation |
|---|---|
| **PERCLOS** | Percentage of eye closure over a rolling time window |
| **EAR / MAR** | Eye and Mouth Aspect Ratios computed from Face Mesh landmarks each frame |
| **Head Slump/Nod** | 3D head pose (via `cv2.solvePnP`) tracked for abnormal pitch changes |
| **Danger Score** | Weighted composite of PERCLOS, yawn frequency, and head-pose deviation |
| **Alerts** | Score crossing Yellow/Red thresholds triggers the flashing banner + async buzzer thread |
| **Incident Logging** | Each sleep/yawn event timestamped and appended, exportable via `csv` module |

---

## 📈 Roadmap

- [ ] In-cabin distraction detection (phone use, looking away)
- [ ] Calibration profile per driver
- [ ] Cloud/telematics incident sync
- [ ] Mobile companion alert app

---

<div align="center">

### ⭐ Star this repo if the Drowsiness Detector kept you safe!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14B8A6,100:2563EB&height=120&section=footer"/>

</div>
