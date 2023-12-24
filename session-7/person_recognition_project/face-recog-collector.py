import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

xml_file_path = os.path.join(script_dir, '..', '..', 'Datasets', 'haarcascade_frontalface_default.xml')

detector = cv2.CascadeClassifier(xml_file_path)

name = input("Enter your name:")

frames = []
outputs = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            cut = frame[y:y+h,x:x+w]
            fix = cv2.resize(cut,(100,100))
            gray = cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)

        cv2.imshow("My screen",frame)
        cv2.imshow("My face",gray)
    
    key = cv2.waitKey(1)

    if key==ord("q"):
        break

    if key==ord("c"):
        img_path = os.path.join(script_dir,name+".jpeg")
        #cv2.imwrite(img_path, frame)
        frames.append(gray.flatten())
        outputs.append([name])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y,X])
print(data.shape)

fname = os.path.join(script_dir,"face_data.npy")

if os.path.exists(fname):
    old = np.load(fname)
    data = np.vstack([old,data])

np.save(fname,data)

cap.release()
cv2.destroyAllWindows()