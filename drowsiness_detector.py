"""
===============================================================================
       SENTINEL DRIVE - OpenCV Driver Drowsiness & Fatigue Detector
===============================================================================
Real-time Computer Vision system using OpenCV and MediaPipe / Haar Cascades.
Calculates Eye Aspect Ratio (EAR) for microsleep and Mouth Aspect Ratio (MAR) for yawning.
"""

import cv2
import time
import math
import csv
import os
import platform
import numpy as np

# Audio Alert Handling
def trigger_alert_sound():
    """Play warning alarm tone using OS native sound engine."""
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(2200, 300)  # Frequency: 2200Hz, Duration: 300ms
        except Exception:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)

# Facial Landmark Index Constants
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = {"left": 61, "right": 291, "top": 13, "bottom": 14}

def euclidean_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_ear(eye_pts):
    """Calculate Eye Aspect Ratio (EAR)."""
    v1 = euclidean_dist(eye_pts[1], eye_pts[5])
    v2 = euclidean_dist(eye_pts[2], eye_pts[4])
    h = euclidean_dist(eye_pts[0], eye_pts[3])
    if h == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_dict):
    """Calculate Mouth Aspect Ratio (MAR) for Yawning."""
    v = euclidean_dist(mouth_dict["top"], mouth_dict["bottom"])
    h = euclidean_dist(mouth_dict["left"], mouth_dict["right"])
    if h == 0:
        return 0.2
    return v / h

def draw_hud_bar(img, label, value, max_val, thresh, x, y, w, h, color):
    """Draw custom HUD gauge bar directly on OpenCV frame."""
    cv2.putText(img, f"{label}: {value:.2f}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (241, 245, 249), 1)
    # Background bar
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 41, 59), -1)
    # Fill progress bar
    fill_w = int(min(1.0, value / max_val) * w)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    # Threshold indicator line
    thresh_x = x + int((thresh / max_val) * w)
    cv2.line(img, (thresh_x, y - 2), (thresh_x, y + h + 2), (0, 0, 255), 2)

def run_opencv_detector():
    print("=" * 70)
    print("      SENTINEL DRIVE - OpenCV DRIVER DROWSINESS DETECTOR      ")
    print("=" * 70)
    print("Initializing OpenCV Camera Stream...")

    # Load MediaPipe FaceMesh safely
    face_mesh = None
    has_mediapipe = False
    try:
        import mediapipe as mp
        mp_fm = None
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            mp_fm = mp.solutions.face_mesh
        else:
            try:
                import mediapipe.python.solutions.face_mesh as mp_fm
            except Exception:
                pass

        if mp_fm is not None:
            face_mesh = mp_fm.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            has_mediapipe = True
            print("[INFO] MediaPipe High-Precision Facial Landmark Mesh: LOADED")
    except Exception as e:
        print(f"[INFO] MediaPipe Mesh fallback mode: {e}")

    if not has_mediapipe:
        print("[INFO] Running in Pure OpenCV Haar Cascades Detector Mode.")

    # Load OpenCV Haar Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    # Video Capture setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access OpenCV Video Capture (Camera index 0).")
        print("[TIP] Ensure your webcam is connected or use the Web Dashboard at http://localhost:8080")
        return

    # Detection Thresholds
    ear_threshold = 0.22
    mar_threshold = 0.55
    trigger_frames = 18

    # State Counters
    drowsy_counter = 0
    yawn_counter = 0
    total_drowsy_alarms = 0
    total_yawns = 0
    
    # CSV Log initialization
    csv_file = "drowsiness_incidents_log.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Detector Engine", "Event Type", "EAR", "MAR", "Frames"])

    prev_time = time.time()

    print("[SUCCESS] OpenCV Detector Running! Press 'q' to quit, 'r' to reset stats, 's' to screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Blank frame received from camera.")
            break

        # Mirror flip
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        status_text = "STATUS: DRIVER ALERT"
        status_color = (157, 255, 0)

        ear_val = 0.30
        mar_val = 0.18

        if has_mediapipe and face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye_pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in LEFT_EYE_INDICES]
                right_eye_pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in RIGHT_EYE_INDICES]
                
                mouth_dict = {
                    "left": (int(landmarks[MOUTH_INDICES["left"]].x * w), int(landmarks[MOUTH_INDICES["left"]].y * h)),
                    "right": (int(landmarks[MOUTH_INDICES["right"]].x * w), int(landmarks[MOUTH_INDICES["right"]].y * h)),
                    "top": (int(landmarks[MOUTH_INDICES["top"]].x * w), int(landmarks[MOUTH_INDICES["top"]].y * h)),
                    "bottom": (int(landmarks[MOUTH_INDICES["bottom"]].x * w), int(landmarks[MOUTH_INDICES["bottom"]].y * h)),
                }

                ear_left = calculate_ear(left_eye_pts)
                ear_right = calculate_ear(right_eye_pts)
                ear_val = (ear_left + ear_right) / 2.0
                mar_val = calculate_mar(mouth_dict)

                # Draw OpenCV Contour Visualizations
                eye_color = (0, 0, 255) if ear_val < ear_threshold else (0, 255, 157)
                cv2.polylines(frame, [np.array(left_eye_pts)], True, eye_color, 2)
                cv2.polylines(frame, [np.array(right_eye_pts)], True, eye_color, 2)

                mouth_color = (0, 183, 255) if mar_val > mar_threshold else (255, 243, 0)
                cv2.line(frame, mouth_dict["top"], mouth_dict["bottom"], mouth_color, 2)
                cv2.line(frame, mouth_dict["left"], mouth_dict["right"], mouth_color, 2)
            else:
                status_text = "STATUS: NO FACE DETECTED"
                status_color = (0, 165, 255)

        else:
            # Pure OpenCV Haar Cascade Detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 243, 255), 2)
                
                roi_gray = gray[fy:fy + fh, fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) < 2:
                    ear_val = 0.15
                else:
                    ear_val = 0.30

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 255, 157), 1)
            else:
                status_text = "STATUS: NO FACE DETECTED"
                status_color = (0, 165, 255)

        # Drowsiness Logic
        if ear_val < ear_threshold:
            drowsy_counter += 1
            if drowsy_counter >= trigger_frames:
                status_text = "🚨 DROWSINESS ALARM! WAKE UP!"
                status_color = (85, 0, 255)
                trigger_alert_sound()

                if drowsy_counter == trigger_frames:
                    total_drowsy_alarms += 1
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, "OpenCV Engine", "CRITICAL DROWSINESS", f"{ear_val:.2f}", f"{mar_val:.2f}", drowsy_counter])
        else:
            drowsy_counter = max(0, drowsy_counter - 1)

        # Yawning Logic
        if mar_val > mar_threshold:
            yawn_counter += 1
            if yawn_counter == 12:
                total_yawns += 1
                status_text = "⚠️ YAWN DETECTED (FATIGUE)"
                status_color = (0, 183, 255)
        else:
            yawn_counter = max(0, yawn_counter - 1)

        # Draw OpenCV HUD Visuals
        cv2.rectangle(frame, (0, 0), (w, 40), (11, 15, 25), -1)
        cv2.putText(frame, "SENTINEL DRIVE - OpenCV Engine", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 243, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        draw_hud_bar(frame, "EAR", ear_val, 0.40, ear_threshold, 15, 60, 180, 14, (0, 243, 255) if ear_val >= ear_threshold else (0, 0, 255))
        draw_hud_bar(frame, "MAR", mar_val, 0.80, mar_threshold, 210, 60, 180, 14, (0, 183, 255) if mar_val > mar_threshold else (0, 255, 157))

        cv2.rectangle(frame, (0, h - 45), (w, h), (11, 15, 25), -1)
        cv2.putText(frame, status_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(frame, f"Alarms: {total_drowsy_alarms} | Yawns: {total_yawns}", (w - 260, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Render Frame
        cv2.imshow("OpenCV Driver Drowsiness System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Exiting OpenCV Detector.")
            break
        elif key == ord('r'):
            drowsy_counter = 0
            yawn_counter = 0
            total_drowsy_alarms = 0
            total_yawns = 0
            print("[INFO] Statistics reset.")
        elif key == ord('s'):
            filename = f"drowsiness_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved snapshot: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_opencv_detector()
