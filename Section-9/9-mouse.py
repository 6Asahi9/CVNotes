import cv2
import numpy as np
import ht_Module as htm
import time 
import autopy

wcam, hcam = 640, 480
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

ptime = 0

smoothness = 7.7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetection(maxHands=1)

widthScreen , HeightScreen = autopy.screen.size()

while True:
    success, img = cap.read()
    if not success: break

    img = detector.findHands(img)
    lmlist, bbox = detector.findPositions(img)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:] #index
        x2, y2 = lmlist[12][1:] #middle

        fingers = detector.fingerup()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            # print("moving mode")

            x3 = np.interp(x1, (frameR, wcam - frameR), (0, widthScreen))
            y3 = np.interp(y1, (frameR, hcam - frameR), (0, HeightScreen))

            #since it shakes a lot we need to smoothen it
            clocX = plocX + (x3 - plocX) / smoothness 
            clocY = plocY + (y3 - plocY) / smoothness

            autopy.mouse.move(widthScreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY
        
        if fingers[1] == 1 and fingers[2] == 1:
            if len(lmlist) > 12:  # Ensure index (8) and middle (12) are in lmlist
                length, img, info = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (info[4], info[5]), 15, (0, 0, 255), cv2.FILLED)
                    
                    autopy.mouse.click()
                
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)