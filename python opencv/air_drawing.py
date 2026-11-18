import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded!")

print("Initializing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Try changing the camera index (try 1, 2, etc. instead of 0)")
    exit(1)

print("Camera opened successfully!")

canvas = None
drawing_color = (0, 255, 0)
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
color_names = ['Green', 'Red', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White']

prev_x, prev_y = None, None

def get_finger_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    fingers = []
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    if hand_landmarks[4].x > hand_landmarks[3].x:
        fingers.insert(0, 1)
    else:
        fingers.insert(0, 0)
    return fingers

def is_in_palette(x, y, w, h):
    return x < 100 and y < len(colors) * 50 + 50

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

print("Initializing hand landmarker...")
try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
    print("Hand landmarker initialized!")
    
    frame_timestamp_ms = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame!")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33
        
        for i, color in enumerate(colors):
            cv2.rectangle(frame, (10, 10 + i * 50), (90, 50 + i * 50), color, -1)
            cv2.putText(frame, color_names[i], (15, 35 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                fingers = get_finger_up(hand_landmarks)
                
                index_tip = hand_landmarks[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                
                if is_in_palette(x, y, w, h):
                    color_idx = (y - 10) // 50
                    if 0 <= color_idx < len(colors):
                        drawing_color = colors[color_idx]
                else:
                    if sum(fingers) == 1 and fingers[1] == 1:
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, 10)
                        prev_x, prev_y = x, y
                    elif sum(fingers) == 2 and fingers[1] == 1 and fingers[2] == 1:
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 30)
                        prev_x, prev_y = x, y
                    elif sum(fingers) == 5:
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        prev_x, prev_y = None, None
                    else:
                        prev_x, prev_y = None, None
        
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        
        cv2.imshow('Air Drawing App', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Quitting app...")
            break
        elif key == ord('s'):
            cv2.imwrite('air_drawing.png', canvas)
            print('Drawing saved as air_drawing.png')
except Exception as e:
    print(f"ERROR: {e}")
finally:
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
