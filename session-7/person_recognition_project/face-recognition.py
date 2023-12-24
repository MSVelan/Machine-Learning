import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(script_dir,"face_data.npy")
data = np.load(img_path)
X = data[:,1:].astype(np.uint8)
y = data[:,0]


model = KNeighborsClassifier()
model.fit(X,y)

xml_file_path = os.path.join(script_dir, '..', '..', 'Datasets', 'haarcascade_frontalface_default.xml')

detector = cv2.CascadeClassifier(xml_file_path)


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

            out = model.predict([gray.flatten()])

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,8e-1,(255,0,0),2)

            cv2.imshow("My face",gray)
        cv2.imshow("My screen",frame)
        
    
    key = cv2.waitKey(1)

    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()