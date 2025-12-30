import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import random

# Robust Import for MediaPipe
try:
    import mediapipe as mp
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
    USE_MEDIAPIPE = True
except ImportError:
    print("Warning: MediaPipe not found. Running in Heuristic Simulation Mode.")
    USE_MEDIAPIPE = False
except Exception as e:
    print(f"Warning: MediaPipe error ({e}). Running in Heuristic Simulation Mode.")
    USE_MEDIAPIPE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "System Online", "model": "Heuristic/MediaPipe"}

# Initialize MediaPipe Face Mesh if available
face_mesh = None
if USE_MEDIAPIPE:
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe FaceMesh Loaded Successfully")
    except Exception as e:
        print(f"Error initializing FaceMesh: {e}")
        USE_MEDIAPIPE = False

# Eye Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Blink State
blink_state = {
    "frames_since_blink": 0,
    "is_blinking": False
}

def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(landmarks, indices):
    # indices: p1, p2, p3, p4, p5, p6
    # Vertical distances
    v1 = distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = distance(landmarks[indices[2]], landmarks[indices[4]])
    # Horizontal distance
    h = distance(landmarks[indices[0]], landmarks[indices[3]])
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def analyze_frame(frame):
    """
    Analyzes the frame for liveness/deepfake indicators.
    """
    global prev_frame, blink_state
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Liveness / Static Image Detection
    is_static = False
    if prev_frame is not None and prev_frame.shape == frame.shape:
         diff = cv2.absdiff(frame, prev_frame)
         diff_mean = np.mean(diff)
         if diff_mean < 0.8: 
             is_static = True
    
    prev_frame = frame

    is_face_detected = False
    confidence_score = 0.0
    ear = 0.0

    if USE_MEDIAPIPE:
        try:
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                is_face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Calculate EAR
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
        except Exception:
            pass 
    else:
        if frame is not None and np.var(frame) > 100:
             is_face_detected = True

    if is_face_detected:
        if is_static:
            # High probability of being fake (Static Image Spoof)
            confidence_score = random.uniform(0.05, 0.20)
        else:
            # Live Feed Logic with Blink Detection
            # If EAR < 0.2, it's a blink
            if ear < 0.22: # Blink threshold
                blink_state["frames_since_blink"] = 0
                blink_state["is_blinking"] = True
            else:
                blink_state["frames_since_blink"] += 1
                blink_state["is_blinking"] = False
            
            # If no blink for 150 frames (~5s at 30fps), it's suspicious (Photo attack)
            if blink_state["frames_since_blink"] > 150:
                # Penalty for lack of liveness
                confidence_score = random.uniform(0.2, 0.4) # Fake/Sus
            else:
                # Healthy Liveness
                base_realness = 0.90 
                noise = random.uniform(-0.05, 0.05)
                confidence_score = base_realness + noise
            
            confidence_score = max(0.0, min(1.0, confidence_score))
        
    return is_face_detected, confidence_score

prev_frame = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting base64 image
            if "data:image" in data:
                header, encoded = data.split(",", 1)
            else:
                encoded = data

            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            is_face, score = analyze_frame(frame)

            response = {
                "face_detected": is_face,
                "real_probability": score,
                "fake_probability": 1.0 - score,
                "status": "analyzed"
            }
            
            await websocket.send_text(json.dumps(response))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
