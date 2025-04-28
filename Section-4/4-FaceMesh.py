import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)
mpface = mp.solutions.face_mesh
face = mpface.FaceMesh()
mpdraw = mp.solutions.drawing_utils

ptime = 0
while True:
    success, img = cap.read()
    if not success:
        break
    img_resized = cv2.resize(img, (640, 480))
    imgRGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    results = face.process(imgRGB)
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         for landmark in face_landmarks.landmark:
    #             print(f"Landmark - x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

    # if results.multi_face_landmarks:
    #     for facelms in results.multi_face_landmarks:
    #         mpdraw.draw_landmarks(img_resized, facelms, mpface.FACEMESH_TESSELATION)
    
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            # Define custom color for landmarks and connections
            landmark_drawing_spec = mpdraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green color
            connection_drawing_spec = mpdraw.DrawingSpec(color=(255, 0, 0), thickness=2)  # Blue color

            # Draw landmarks and connections with the custom specs
            mpdraw.draw_landmarks(
                img_resized, 
                facelms, 
                mpface.FACEMESH_CONTOURS,  # Connections (lines)
                landmark_drawing_spec,     # Landmarks
                connection_drawing_spec    # Connections
            )

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("face", img_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
