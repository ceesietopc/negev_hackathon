#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os

def eyes_detect():

    face_cascade = "{base_path}/haarcascades/haarcascade_frontalface_alt.xml".format(
        base_path=os.path.abspath(os.path.dirname(__file__)))

    eye_cascade = "{base_path}/haarcascades/haarcascade_eye.xml".format(
        base_path=os.path.abspath(os.path.dirname(__file__)))

    mouth_cascade = "{base_path}/haarcascades/haarscascade_mcs_mouth".format(
        base_path=os.path.abspath(os.path.dirname(__file__)))

    faceCascade = cv2.CascadeClassifier(face_cascade)
    mouthCascade = cv2.CascadeClassifier(mouth_cascade)
    eyeCascade = cv2.CascadeClassifier(eye_cascade)

    cap = cv2.VideoCapture(0)

    ds_factor = 0.5

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray scale', gray)

        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        mouth_rects = mouthCascade.detectMultiScale(gray, 1.7, 11)

        #mouth
        for (x,y,w,h) in mouth_rects:
            y = int(y - 0.15*h)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            break

        for (x,y,w,h) in faces:
            #eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eyeCascade.detectMultiScale(roi_gray)

            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
                radius = int(0.3 * (w_eye + h_eye))
                color = (0, 255, 0)
                thickness = 3
                cv2.circle(roi_color, center, radius, color, thickness)

        cv2.imshow('Eye detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    eyes_detect()
