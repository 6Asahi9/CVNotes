import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('HumanMotions/MultiplePeople.mp4')
ptime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose() #model
mpDraw = mp.solutions.drawing_utils #lines

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    #? print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h, w, c = img.shape
        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)

            #custom landmark
            # if id == 0:
            #     cv2.circle(img, (cx, cy), 25, (225, 0, 225), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0,0), 3)
    
    cv2.imshow("Movement Tracking", img)

    cv2.waitKey(1)