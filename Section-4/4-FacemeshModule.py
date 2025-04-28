# facemesh_module.py

import cv2
import mediapipe as mp
import time

class FaceMeshModule:
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.width = width
        self.height = height
        self.p_time = 0

        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh()
        self.mp_draw = mp.solutions.drawing_utils

        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)

    def process_frame(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self.face.process(img_rgb)

        # Uncomment to print all landmark coordinates
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         for landmark in face_landmarks.landmark:
        #             print(f"Landmark - x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

        # Uncomment to use tesselation lines instead of contours
        # if results.multi_face_landmarks:
        #     for facelms in results.multi_face_landmarks:
        #         self.mp_draw.draw_landmarks(resized, facelms, self.mp_face.FACEMESH_TESSELATION)

        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    resized,
                    facelms,
                    self.mp_face.FACEMESH_CONTOURS,
                    self.landmark_drawing_spec,
                    self.connection_drawing_spec
                )

        return resized

    def update_fps(self):
        c_time = time.time()
        fps = 1 / (c_time - self.p_time)
        self.p_time = c_time
        return int(fps)

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            processed = self.process_frame(frame)
            fps = self.update_fps()

            cv2.putText(processed, f"FPS: {fps}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Face Mesh", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = FaceMeshModule()
    app.run()

if __name__ == "__main__":
    main()
