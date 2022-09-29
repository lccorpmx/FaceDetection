import cv2
import os
import numpy as np
from PIL import Image

if not os.path.exists('Rostros encontrados'):
    print('Carpeta creada: Rostros encontrados')
    os.makedirs('Rostros encontrados')
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 0
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray_image, 1.3, 5)
    k = cv2.waitKey(1)
    #if k == 27:
    #    break

    if k == 27:
        break

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('rostro',rostro)
        lista=np.array(rostro) #convierto la imagen a un array
        cv2.imwrite(f'Rostros encontrados/rostro_{count}.jpg',lista)
        count = count +1

    cv2.imshow('frame',frame)
    
for i in lista:
    #imageTransform=np.array(i)
    print(type(i))



cap.release()
