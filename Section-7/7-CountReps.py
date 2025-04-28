import cv2
import mediapipe as mp

# Setup
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Rep counts & states
arm_count = 0
squat_count = 0
arm_state = 'down'
squat_state = 'up'

arm_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror mode
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    h, w, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # === ARM REP LOGIC ===
        Relbow_y = landmarks[14].y
        Rhand_y = landmarks[16].y

        if arm_state is None:
            if Relbow_y < Rhand_y:
                arm_state = 'up'
            else:
                arm_state = 'down'

        if Relbow_y < Rhand_y and arm_state == 'down':
            arm_count += 1
            arm_state = 'up'
        elif Relbow_y > Rhand_y and arm_state == 'up':
            arm_state = 'down'

        # === SQUAT REP LOGIC ===
        hip_y = landmarks[24].y
        knee_y = landmarks[26].y

        if hip_y >= knee_y + 0.05 and  squat_state == 'up':
            squat_count += 1
            squat_state = 'down'
        elif hip_y < knee_y:
            squat_state = 'up'

        # === Draw Landmarks (Optional but useful) ===
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Debug dots (right wrist, elbow, hip, knee)
        # for id in [16, 14, 24, 26]:
        #     cx = int(landmarks[id].x * w)
        #     cy = int(landmarks[id].y * h)
        #     cv2.circle(frame, (cx, cy), 8, (255, 0, 255), -1)

    # === Display Reps ===
    cv2.putText(frame, f'Arms: {arm_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Squats: {squat_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Full Body Rep Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
