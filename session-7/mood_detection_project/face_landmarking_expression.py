import cv2
import os
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()

script_dir = os.path.dirname(os.path.abspath(__file__))

data_file_path = os.path.join(script_dir, '..', '..', 'Datasets', 'shape_predictor_68_face_landmarks.dat')

predictor = dlib.shape_predictor(data_file_path)

mood = input("Enter your mood:")
frames = []
outputs = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray,face)
        parts = landmarks.parts()
        """
        lip_up = parts[62].y
        lip_down = parts[66].y
        if lip_down - lip_up>5:
            print("Mouth is open")
            # instead of this we can also control the keyboard by 
        else:
            print("Mouth is closed")

        """
        
        expression = np.array([[point.x-face.left(),point.y-face.top()] for point in parts[17:]]) 

        for point in landmarks.parts():
            cv2.circle(frame,(point.x,point.y),2,(255,0,0),1)

    # print(faces)
    if ret:
        cv2.imshow("My screen",frame)
    
    key = cv2.waitKey(1)

    if key==ord("q"):
        break
    elif key==ord("c"):
        frames.append(expression.flatten())
        outputs.append([mood])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y,X])
print(data.shape)

fname = os.path.join(script_dir,"mood_detection_data.npy")

if os.path.exists(fname):
    old = np.load(fname)
    data = np.vstack([old,data])

np.save(fname,data)

cap.release()
cv2.destroyAllWindows()