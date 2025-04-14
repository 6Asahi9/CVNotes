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
        return img

def main():
    cap = cv2.VideoCapture('HumanMotions/MultiplePeople.mp4')
    detector = PoseDetector()
    ptime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Movement Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
