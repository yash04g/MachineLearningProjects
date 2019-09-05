import cv2
import numpy as np

file_name = input("Enter the name of Student ")
dataset_path = "./data/"

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

face_data = []
skip = 0

while True:
    ret,frame = cap.read()
    if ret==False:
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) 
    faces = sorted(faces,key=lambda f: f[2]*f[3],reverse = True)

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Extracting (Cropping out required face) : Region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w-offset]
        face_section = cv2.resize(face_section,(100,100))
        skip += 1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    
    cv2.imshow("Camera Feed",frame)
    cv2.imshow("Face Section",face_section)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
# Converting our face_data list to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path+file_name+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()

