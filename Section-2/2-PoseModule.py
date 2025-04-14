import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self, mode=False, smoothness=True, detectionConf=0.5, trackingconf=0.5):
        self.mode = mode
        self.smoothness = smoothness
        self.detectionconf = detectionConf
        self.trackingconf = trackingconf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smoothness,
            min_detection_confidence=self.detectionconf,
            min_tracking_confidence=self.trackingconf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return results.pose_landmarks

    def getPosition(self, img, draw=False):
        landmarks = self.findPose(img, draw)
        positions = []
        if landmarks:
            h, w, c = img.shape
            for id, lm in enumerate(landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                positions.append((id, cx, cy))
                # if draw:
                #     cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return positions

def main():
    cap = cv2.VideoCapture('HumanMotions/MultiplePeople.mp4')  # Replace with your video or camera source
    ptime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        positions = detector.getPosition(img, draw=True)  # Get landmark positions
        for id, x, y in positions:
            print(f"ID: {id}, Position: ({x}, {y})")

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Movement Tracking", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()
