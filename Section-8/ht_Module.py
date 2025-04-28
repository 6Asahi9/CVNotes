import cv2
import mediapipe as mp
import time
import math

class handDetection():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)  # Ensure float type
        self.trackCon = float(trackCon)          # Ensure float type

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils 

        self.tipIds = [4, 8, 12, 16, 20]  # Indices for the tips of the thumb, index, middle, ring, and pinky

    def findHands(self, img, draw=True): 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=5),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
        return img
    
    def findPositions(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 15, (225, 0, 225), cv2.FILLED)

            # Now calculate BBOX
            x_list = [pt[1] for pt in self.lmList]
            y_list = [pt[2] for pt in self.lmList]

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            return self.lmList, bbox

        return [], None

    
    def fingerup(self):
        fingers = []
        # print("Length of lmList:", len(self.lmList))
        # print("tipIds:", self.tipIds)
        
        # For thumb 
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # For other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def findDistance(self, p1, p2, img=None, draw=True):
        """
        Find distance between two landmark points
        p1, p2: landmark ids (like 8 and 12)
        img: optional, pass image to draw
        draw: bool, whether to draw or not
        """
        if len(self.lmList) == 0:
            return 0, None, None  # No hand detected
        
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        
        return length, img, (x1, y1, x2, y2, cx, cy)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = handDetection() 
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPositions(img)
        if len(lmlist) != 0:
            print(lmlist[4])
         
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {round(fps, 2)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 3, cv2.LINE_AA, False)

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

if __name__ == "__main__":
    main()
