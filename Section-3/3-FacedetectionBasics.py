import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
ptime = 0

mpface = mp.solutions.face_detection
face = mpface.FaceDetection()
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    
    # If no frame is captured, break the loop
    if not success:
        break
    
    # Convert the frame to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect faces
    results = face.process(imgRGB)

    # Check if faces are detected
    if results.detections:
        for detection in results.detections:
            # Draw the bounding box around the face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw rectangle on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optionally draw the keypoints (for debugging or additional visualization)
            mpdraw.draw_detection(img, detection)
            
            # Print the detection score and bounding box (optional)
            print(detection.score)  # Confidence score
            print(detection.location_data.relative_bounding_box)  # Bounding box info
    
    # Calculate FPS
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    # Display FPS on the image
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #confidence bar on box
    cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2)

    # Show the image with face detection
    cv2.imshow("Face Detection", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
