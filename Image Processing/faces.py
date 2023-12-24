import cv2
import os
cap = cv2.VideoCapture(0)

curdir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(curdir,"haar_cascade_frontal_face_detection_default_xml.xml")
classifier = cv2.CascadeClassifier(filename)

while True:
    ret, frame = cap.read()
    if ret:
        faces = classifier.detectMultiScale(frame)

        for face in faces:
            [x,y,w,h] = face
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

        cv2.imshow("My window",frame)
    key = cv2.waitKey(1)

    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


