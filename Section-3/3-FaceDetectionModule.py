import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.previous_time = 0

    def detect_and_display(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.mp_draw.draw_detection(img, detection)
                
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                            (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (0, 255, 0), 
                            2)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - self.previous_time) if self.previous_time else 0
        self.previous_time = current_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        output = detector.detect_and_display(img)
        cv2.imshow("Face Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
