import cv2
import mediapipe as mp

# Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark groups
FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
    67, 109
]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7, 246]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 390, 249, 466]
TOP_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88]
BOTTOM_LIP = [178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405]
# For Eye Trackingss
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
REGION_INDICES = set(FACE_OUTLINE+ LEFT_EYE +RIGHT_EYE  + 
                     RIGHT_EYEBROW+ LEFT_EYEBROW+ LEFT_IRIS+ RIGHT_IRIS+ TOP_LIP+ BOTTOM_LIP)

# Webcam
cap = cv2.VideoCapture(0)

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx not in REGION_INDICES:
                        h, w, _ = frame.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Selective Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
