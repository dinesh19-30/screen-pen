# 🖐️ Air Pen Drawing using Python (Index Finger Tracking)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🎯 Overview

This project lets you **draw in the air using your index finger** — no mouse, no stylus, just your webcam 🎥.  
It uses **OpenCV** and **MediaPipe** to detect your hand in real-time and convert your **index finger tip** into a virtual pen.

You can clear, save, or quit easily using keyboard shortcuts.

---

## 🧠 Demo

✋ **Raise your index finger** → Start drawing  
🤚 **Lower your index finger** → Stop drawing  
💾 **Press `s`** → Save drawing  
🧹 **Press `c`** → Clear the canvas  
❌ **Press `q`** → Quit the app  

*(Optional: Add your own GIF or screenshot here)*

---

## ⚙️ Features

✅ Real-time drawing using your index finger  
✅ “Air Pen Mode” – draws only when the finger is raised  
✅ Smooth tracking with exponential moving average  
✅ Save drawings as images  
✅ Lightweight (runs on CPU)

---

## 🧩 Requirements

You need Python 3.8+ and the following libraries:


pip install opencv-python mediapipe numpy
🚀 Run the App
Save the script below as air_pen_draw.py and run:

bash
Copy code
python air_pen_draw.py
Then, raise your index finger to draw in the air!

💻 Full Code


```
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

MAX_POINTS = 1024
SMOOTHING = 0.6
BRUSH_THICKNESS = 6
BRUSH_COLOR = (0, 0, 255)
ERASE_KEY = ord('c')
SAVE_KEY = ord('s')
QUIT_KEY = ord('q')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist_x = lm[0].x
    thumb_tip_x = lm[4].x
    thumb_ip_x = lm[3].x
    fingers = [thumb_tip_x < thumb_ip_x if wrist_x < 0.5 else thumb_tip_x > thumb_ip_x]
    for id in [8, 12, 16, 20]:
        fingers.append(lm[id].y < lm[id - 2].y)
    return tuple(fingers)

def scaled_point(lm, frame_w, frame_h):
    return int(lm.x * frame_w), int(lm.y * frame_h)

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    if not cap.isOpened():
        print("Unable to open webcam.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Can't read from webcam.")
        return
    h, w = frame.shape[:2]
    canvas = np.zeros_like(frame)
    last_x, last_y = None, None
    stroke_pts = deque(maxlen=MAX_POINTS)
    prev_point = None
    smoothing_point = None

  while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        draw_mode = False
        cursor = None

 if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            thumb, index, middle, ring, pinky = fingers_up(hand_landmarks)
            ix, iy = scaled_point(lm[8], w, h)
            cursor = (ix, iy)
            if index and not middle:
                draw_mode = True
            else:
                draw_mode = False
             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

   if cursor is not None:
            if smoothing_point is None:
                smoothing_point = cursor
            else:
                sx = int(smoothing_point[0] * (1 - SMOOTHING) + cursor[0] * SMOOTHING)
                sy = int(smoothing_point[1] * (1 - SMOOTHING) + cursor[1] * SMOOTHING)
                smoothing_point = (sx, sy)
            px, py = smoothing_point
            if draw_mode:
                if prev_point is None:
                    prev_point = (px, py)
                    stroke_pts.append(prev_point)
                else:
                    dx = px - prev_point[0]
                    dy = py - prev_point[1]
                    if (dx * dx + dy * dy) > 2:
                        stroke_pts.append((px, py))
                        prev_point = (px, py)
            else:
                if len(stroke_pts) > 1:
                    pts = np.array(stroke_pts, dtype=np.int32)
                    cv2.polylines(canvas, [pts], False, BRUSH_COLOR, BRUSH_THICKNESS, cv2.LINE_AA)
                    stroke_pts.clear()
                prev_point = None
            cv2.circle(frame, (px, py), 8, (0, 255, 0) if draw_mode else (255, 0, 0), -1)

   preview = canvas.copy()
        if len(stroke_pts) > 1:
            pts = np.array(stroke_pts, dtype=np.int32)
            cv2.polylines(preview, [pts], False, BRUSH_COLOR, BRUSH_THICKNESS, cv2.LINE_AA)

   alpha = 0.6
        combined = cv2.addWeighted(frame, 1.0, preview, alpha, 0)
        cv2.putText(combined, "Draw: index up | c: clear | s: save | q: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow("Air Pen Drawing", combined)

   key = cv2.waitKey(1) & 0xFF
        if key == QUIT_KEY:
            break
        elif key == ERASE_KEY:
            canvas[:] = 0
            stroke_pts.clear()
            prev_point = None
            smoothing_point = None
            print("Canvas cleared.")
        elif key == SAVE_KEY:
            fname = f"drawing_{int(time.time())}.png"
            white_bg = 255 * np.ones_like(canvas)
            saved = cv2.addWeighted(white_bg, 1.0, canvas, 1.0, 0)
            cv2.imwrite(fname, saved)
            print(f"Saved {fname}")
cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
```
## 🧾 Keyboard Shortcuts
Key	Action
c	Clear the canvas
s	Save current drawing
q	Quit the app

## 🧠 How It Works
Detects hand landmarks using MediaPipe

Identifies when the index finger is raised

Draws lines between frames where the finger moves

Blends the drawing over the webcam feed using OpenCV

## 🖼️ Output
Each saved image is named like:

php-template
Copy code
drawing_<timestamp>.png
## 🧠 Tech Stack
Python 3.8+
OpenCV
MediaPipe
NumPy
Webcam input

🧩 Future Enhancements
🎨 Add color selection
🧤 Add eraser gesture
🌈 Multi-finger tools (thickness, color)
💾 Auto-save drawings

## 👨‍💻 Author
Developed by Dinesh19-30
Inspired by the concept of AI Air Drawing — real-time creativity powered by hand-tracking.

## 📜 License
MIT License © 2025 Dinesh19-30
