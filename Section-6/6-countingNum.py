import cv2
import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB (MediaPipe expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get hand landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Drawing the landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Example of checking how many fingers are up
            up_fingers = 0
            if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <= landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                up_fingers += 1
            if landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <= landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                up_fingers += 1
            if landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <= landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                up_fingers += 1
            if landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y <= landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                up_fingers += 1
            if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                up_fingers += 1

            # Display the number of fingers detected
            cv2.putText(image, f"Fingers Up: {up_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
