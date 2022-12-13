"""
For finding the face and face landmarks for further manipulication
"""
import csv
import cv2 as cv
import mediapipe as mp
import numpy as np
from Emotion_Detector import calc_bounding_rect,pre_process_landmark,draw_bounding_rect ,draw_info_text,calc_landmark_list
from unity.keypoint_classifier import KeyPointClassifier
class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.keypoint_classifier = KeyPointClassifier()
        self.facial_emotion_id = 2
    def findFaceMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv.cvtColor(cv.flip(img, 1), cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        img.flags.writeable = False
        self.results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape


        self.faces = []

        use_brect = True

        # Read labels
        with open('keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        mode = 0



        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec)

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])

                    # show the id of each point on the image
                    # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                self.faces.append(face)

                brect = calc_bounding_rect(img, face_landmarks)


                # Landmark calculation
                landmark_list = calc_landmark_list(img, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # emotion classification
                self.facial_emotion_id = self.keypoint_classifier(pre_processed_landmark_list)
                # Drawing part
                img = draw_bounding_rect(use_brect, img, brect)
                img = draw_info_text(
                    img,
                    brect,
                    keypoint_classifier_labels[self.facial_emotion_id])
        return img, self.faces, self.facial_emotion_id


# sample run of the module
def main():

    detector = FaceMeshDetector()

    cap = cv.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces ,emoID= detector.findFaceMesh(img)

        # if faces:
        #     print(faces[0])

        cv.imshow('MediaPipe FaceMesh', img)

        # press "q" to leave
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    # demo code
    main()
