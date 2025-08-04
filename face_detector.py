import cv2
import mediapipe as mp
import math

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye_landmarks):
    ver_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    ver_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    hor_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    if hor_dist == 0: return 0.0
    return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)

def calculate_mar(mouth_landmarks):
    ver_dist = euclidean_distance(mouth_landmarks[0], mouth_landmarks[1])
    hor_dist = euclidean_distance(mouth_landmarks[2], mouth_landmarks[3])
    if hor_dist == 0: return 0.0
    return ver_dist / hor_dist

EAR_THRESHOLD = 0.20  
EAR_CONSEC_FRAMES = 20 
MAR_THRESHOLD = 0.5  

drowsy_counter = 0
is_drowsy = False
is_yawning = False

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [13, 14, 61, 291] 


mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                all_landmarks = results.multi_face_landmarks[0].landmark
                
                left_eye = [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in LEFT_EYE_INDICES]
                right_eye = [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in RIGHT_EYE_INDICES]
                mouth = [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in MOUTH_INDICES]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(mouth)
                
                if avg_ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                    if drowsy_counter >= EAR_CONSEC_FRAMES:
                        is_drowsy = True
                else:
                    drowsy_counter = 0
                    is_drowsy = False

                is_yawning = mar > MAR_THRESHOLD


                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if is_drowsy:
                    cv2.putText(frame, "DROWSY!", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if is_yawning:
                    cv2.putText(frame, "YAWNING!", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                for idx, lm in enumerate(all_landmarks):
                    if idx in LEFT_EYE_INDICES or idx in RIGHT_EYE_INDICES or idx in MOUTH_INDICES:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

            cv2.imshow("Fatigue Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()