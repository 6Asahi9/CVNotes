import cv2
import mediapipe as mp
import time 
import ht_Module as htm
import numpy as np
import math

cap = cv2.VideoCapture(0)
mpdraw = mp.solutions.drawing_utils

ptime = 0
detector = htm.handDetection()

#####################################
#from pycraw

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volr = volume.GetVolumeRange() # (min_volume, max_volume, volume_increment)
# volume.SetMasterVolumeLevel(-20.0, None)

#####################################

minvol = volr[0]
maxvol = volr[1]

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findHands(img)
    lmlist, _ = detector.findPositions(img, draw = False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]

        cv2.circle(img, (x1, y1), 15, (225,0,225), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (225,0,225), cv2.FILLED)

        cv2.line(img, (x1,y1), (x2,y2), (0, 225, 0), 3)

        cx, cy = (x1 + x2)// 2, (y1 + y2) //2
        cv2.circle(img, (cx, cy), 15, (225,0,225), cv2.FILLED)

        # hand range = 50 - 300
        # volume range = -65.25 - 0
        length = math.hypot(x2 -x1, y2 -y1)
        vol = np.interp(length , [50, 230], [minvol, maxvol])
        volume.SetMasterVolumeLevel(vol, None)

        # Convert volume level to percentage for visual bar
        vol_bar = np.interp(length, [50, 230], [400, 150])  # y-pos on screen
        vol_perc = np.interp(length, [50, 230], [0, 100])   # 0% to 100%

        # Draw volume bar background
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)  # outer box
        # Draw filled volume level
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        # Draw volume percentage text
        cv2.putText(img, f'{int(vol_perc)} %', (40, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 2)
        
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0,225,0), cv2.FILLED)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, f"FPS : {int(round(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)