import cv2
import mediapipe as mp
import time

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

    def findHands(self, img, draw=True): 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=5),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
        return img

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
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {round(fps, 2)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 225, 0), 3, cv2.LINE_AA, False)

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

if __name__ == "__main__":
    main()
