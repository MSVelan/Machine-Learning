import cv2
import os
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

script_dir = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(script_dir,"mood_detection_data.npy")
data = np.load(img_path)

X = data[:,1:].astype(np.uint8)
y = data[:,0]


# Handle imbalanced data by over-sampling each class
X_balanced, y_balanced = [], []

unique_classes = np.unique(y)
max_samples = max([len(y[y == c]) for c in unique_classes])

for c in unique_classes:
    X_class = X[y == c]
    y_class = y[y == c]

    # Over-sample the minority classes
    X_over, y_over = resample(X_class, y_class, replace=True, n_samples=max_samples, random_state=42)

    X_balanced.extend(X_over)
    y_balanced.extend(y_over)

X_balanced = np.array(X_balanced)
y_balanced = np.array(y_balanced)

model = KNeighborsClassifier()
model.fit(X_balanced, y_balanced)

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
        parts = landmarks.parts()
        
        expression = np.array([[point.x-face.left(),point.y-face.top()] for point in parts[17:]]) 

        for point in landmarks.parts():
            cv2.circle(frame,(point.x,point.y),2,(255,0,0),1)

        out = model.predict([expression.flatten()])

        x,y,w,h = face.left(),face.top(),face.width(),face.height()
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,8e-1,(255,0,0),2)

    if ret:
        cv2.imshow("My screen",frame)
    
    key = cv2.waitKey(1)

    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()