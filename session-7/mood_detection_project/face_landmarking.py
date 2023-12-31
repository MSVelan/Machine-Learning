import cv2
import os
import dlib

detector = dlib.get_frontal_face_detector()

script_dir = os.path.dirname(os.path.abspath(__file__))

data_file_path = os.path.join(script_dir, '..', '..', 'Datasets', 'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(data_file_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray,face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        # print(nose.x,nose.y)
        for point in landmarks.parts():
            cv2.circle(frame,(point.x,point.y),2,(255,0,0),1)

    # print(faces)
    if ret:
        cv2.imshow("My screen",frame)
    
    key = cv2.waitKey(1)

    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()