import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from scipy.spatial import distance as dist


# Load trained model
print("Loading emotion recognition model...")
MODEL_PATH = 'emotion_model_v3_finetuned.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
CLASS_NAMES = ['angry', 'happy', 'sad', 'surprise']
MODEL_INPUT_SHAPE = (48, 48)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


LEFT_EYE_IDXS = [362, 382, 381, 380, 373, 374]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153]
MOUTH_IDXS = [61, 291, 0, 17] # Left corner, Right corner, Top lip, Bottom lip
SMILE_IDXS = [61, 291, 13, 14] # Left corner, Right corner, Upper lip center, Lower lip center
INNER_LEFT_EYEBROW_IDX = 55
INNER_RIGHT_EYEBROW_IDX = 285
LEFT_EYE_TOP_IDX = 386
RIGHT_EYE_TOP_IDX = 159
MOUTH_CENTER_TOP_IDX = 13
MOUTH_CENTER_BOTTOM_IDX = 14

def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_landmarks):
    A = dist.euclidean(mouth_landmarks[2], mouth_landmarks[3]) # Top to bottom lip
    B = dist.euclidean(mouth_landmarks[0], mouth_landmarks[1]) # Left to right corner
    mar = A / B
    return mar

def get_geometric_emotion(landmarks):
    landmarks_np = np.array(landmarks)
    debug_metrics = {}
    left_eye_pts = landmarks_np[LEFT_EYE_IDXS]
    right_eye_pts = landmarks_np[RIGHT_EYE_IDXS]
    left_ear = calculate_ear(left_eye_pts)
    right_ear = calculate_ear(right_eye_pts)
    avg_ear = (left_ear + right_ear) / 2.0
    debug_metrics['ear_score'] = avg_ear

    mouth_pts = landmarks_np[MOUTH_IDXS]
    mar = calculate_mar(mouth_pts)

    smile_pts = landmarks_np[SMILE_IDXS]
    mouth_width = dist.euclidean(smile_pts[0], smile_pts[1])
    mouth_height = dist.euclidean(smile_pts[2], smile_pts[3])
    smile_ratio = mouth_width / (mouth_height + 1e-6)


    # Sadness Score
    mouth_corner_left = landmarks_np[61]
    mouth_corner_right = landmarks_np[291]
    mouth_center_top = landmarks_np[MOUTH_CENTER_TOP_IDX]
    avg_corner_y = (mouth_corner_left[1] + mouth_corner_right[1]) / 2
    center_lip_y = mouth_center_top[1]
    sadness_score = (avg_corner_y - center_lip_y) / (mouth_height + 1e-6)
    debug_metrics['sad_score'] = sadness_score

    # Angry Score
    left_eyebrow_pt = landmarks_np[INNER_LEFT_EYEBROW_IDX]
    right_eyebrow_pt = landmarks_np[INNER_RIGHT_EYEBROW_IDX]
    left_eye_top_pt = landmarks_np[LEFT_EYE_TOP_IDX]
    right_eye_top_pt = landmarks_np[RIGHT_EYE_TOP_IDX]
    left_dist = left_eye_top_pt[1] - left_eyebrow_pt[1]
    right_dist = right_eye_top_pt[1] - right_eyebrow_pt[1]
    avg_brow_eye_dist = (left_dist + right_dist) / 2
    left_eye_height = dist.euclidean(landmarks_np[386], landmarks_np[374])
    right_eye_height = dist.euclidean(landmarks_np[159], landmarks_np[145])
    avg_eye_height = (left_eye_height + right_eye_height) / 2.0
    angry_score = avg_brow_eye_dist / (avg_eye_height + 1e-6)
    debug_metrics['angry_score'] = angry_score

    
    emotion_text = "Neutral (Geometric)" # default

    if sadness_score > 15.0:
        emotion_text = "Sad (Geometric)"
    elif angry_score > 4.0:
        emotion_text = "Angry (Geometric)"
    elif avg_ear > 0.33 and mar > 0.5:
        emotion_text = "Surprise (Geometric)"
    elif smile_ratio > 3.5 and mar > 0.1:
        emotion_text = "Happy (Geometric)"

    return emotion_text, debug_metrics

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame.flags.writeable = True

    # fatigue_status = "Status: Awake"
    cnn_emotion = "CNN Emotion: N/A"
    geo_emotion = "Geometric Emotion: N/A"
    fatigue_status = "Status: No Face"
    debug_metrics = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append((int(lm.x * frame_width), int(lm.y * frame_height)))

            left_eye_pts = np.array([landmarks[i] for i in LEFT_EYE_IDXS])
            right_eye_pts = np.array([landmarks[i] for i in RIGHT_EYE_IDXS])
            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear > 1.6:
                 fatigue_status = "Status: Blinking / Eyes Closed"
            else:
                 fatigue_status = "Status: Awake"

            geo_emotion_text, debug_metrics = get_geometric_emotion(landmarks)

            all_x = [lm[0] for lm in landmarks]
            all_y = [lm[1] for lm in landmarks]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(frame_width, x_max + padding)
            y_max = min(frame_height, y_max + padding)
            if x_max > x_min and y_max > y_min:
                face_crop = frame[y_min:y_max, x_min:x_max]
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, MODEL_INPUT_SHAPE)
                
                input_face = np.expand_dims(resized_face, axis=-1)
                input_face = np.expand_dims(input_face, axis=0)

                prediction = model.predict(input_face, verbose=0)
                pred_index = np.argmax(prediction)
                pred_confidence = prediction[0][pred_index]
                cnn_emotion = f"CNN Emotion: {CLASS_NAMES[pred_index]} ({pred_confidence:.0%})"
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


    cv2.putText(frame, fatigue_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, cnn_emotion, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, geo_emotion_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    sad_score = debug_metrics.get('sad_score', 0)
    angry_score = debug_metrics.get('angry_score', 0)
    cv2.putText(frame, f"Sad Score: {sad_score:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Angry Score: {angry_score:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    ear_score = debug_metrics.get('ear_score', 0) 
    cv2.putText(frame, f"EAR Score: {ear_score:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow('Real-Time Facial Analysis', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()