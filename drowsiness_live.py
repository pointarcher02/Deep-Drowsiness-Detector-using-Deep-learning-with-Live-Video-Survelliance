import winsound
import numpy as np
import cv2
from tensorflow.keras.models import load_model

frequency=2500
duration=1500
path = "haarcascade_frontalface_default.xml"

#here we have set the haarcascade cascade classifier for the frontal face detections
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,5)
counter=0

while True:
    ret,frame=cap.read()
    #now one point we need to note over here is that eye_cascade is an eye detector classifier
    eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(gray_frame,1.1,4)


    for x,y,w,h in eyes:
        roi_gray=gray_frame[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        eyess=eye_cascade.detectMultiScale(roi_gray)
        if len(eyess)==0:
            print("Eyes not detected")
        else:
            for ex,ey,eh,ew in eyess:
                eyes_roi=roi_color[ey:ey+eh,ex:ex+ew]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if(faceCascade.empty()==False):
        print("detected")
    faces=faceCascade.detectMultiScale(gray, 1.1, 4)

    #draw a rectable around the eyes
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font = cv2.FONT_HERSHEY_SIMPLEX


    final_image = cv2.resize(eyes_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0

    eyeNet=load_model("mask_detector.model")
    prediction=eyeNet.predict(final_image)

    if (prediction>=0.3):
        status="Open eyes"
        cv2.putText(frame,status,(150,150),font,3,(0,255,0),2,cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0,0,0), -1)
        #Adding text
        cv2.putText(frame, 'active', (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        
    elif (prediction<0.3):
        counter=counter+1
        status = "Closed Eyes"
        cv2.putText(frame,status,(150,150),font, 3,(0, 0, 255),2,cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame, (x1,y1), (x1 + w1, y1 + h1), (0,0,255), 2)

        if (counter>10):
            x1,y1,w1,h1 = 0,0,175,75
            #Draw black background rectangle to indicate that something is different
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0,0,0), -1)
            #Adding text
            cv2.putText(frame, "Sleep Alert !!!", (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            winsound.Beep(frequency,duration)
            counter=0
        cv2.imshow("Drowsiness detection",frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()