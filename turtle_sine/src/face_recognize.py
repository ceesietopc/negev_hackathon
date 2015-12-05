#!/usr/bin/env python

import sys
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import imutils
import cv2, cv
import uuid
from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
import time



# globalna definicja
pathdir = "data/"

model = PredictableModel(Fisherfaces(), NearestNeighbor())

cascPath = "{base_path}/haarcascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

faceCascade = cv2.CascadeClassifier(cascPath)
print faceCascade


# img
def read_images(path, sz=(256, 256)):
    c = 0
    X, y = [], []
    folder_names = []

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c + 1
    return [X, y, folder_names]

# glowna
def main():
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture(0)
    ds_factor = 0.5

    # inicjalizacja
    pytanie = int(raw_input('How many people are in front of the camera ? \n ilosc:'))

    for i in range(pytanie):
        nome = raw_input('Part of person No. ' + str(i + 1) + ' Whats your name?\n imie:')
        if not os.path.exists(pathdir + nome): os.makedirs(pathdir + nome)

        print ('Are you ready?\n')

        print ('This will take 10 seconds. \n Nacisnij "s" gdy juz bedziesz gotowy!')

        while True:
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 3)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Rozpoznawanie', frame)

            if cv2.waitKey(10) == ord('s'):
                break
        cv2.destroyAllWindows()

        start = time.time()
        count = 0

        while int(time.time() - start) <= 14:

            ret, frame = video_capture.read()
            frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 3)

            for (x, y, w, h) in faces:
                cv2.putText(frame, 'Zapisuje!', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 3, 1)
                count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                resized_image = cv2.resize(frame[y:y + h, x:x + w], (273, 273))

                if count % 5 == 0:
                    print  pathdir + nome + str(time.time() - start) + '.jpg'
                    cv2.imwrite(pathdir + nome + '/' + str(time.time() - start) + '.jpg', resized_image)
            cv2.imshow('Rozpoznawanie', frame)
            cv2.waitKey(10)
        cv2.destroyAllWindows()

    [X, y, subject_names] = read_images(pathdir)
    list_of_labels = list(xrange(max(y) + 1))

    subject_dictionary = dict(zip(list_of_labels, subject_names))
    model.compute(X, y)

    while True:
        # klatka-po-klatce
        ret, frame = video_capture.read()

        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_CUBIC)

        img = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # jezeli znajdziesz morde
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(frame, "Wykryto twarzy: {}".format(len(faces)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            sampleImage = gray[y:y + h, x:x + w]
            sampleImage = cv2.resize(sampleImage, (256, 256))

            [predicted_label, generic_classifier_output] = model.predict(sampleImage)

            print [predicted_label, generic_classifier_output]

            if int(generic_classifier_output['distances']) <= 700:
                # wypisz imie
                cv2.putText(img, 'You are: ' + str(subject_dictionary[predicted_label]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 250), 3, 1)

                rosf = str(subject_dictionary[predicted_label])
                print 'test test %s', (rosf)
                    
            print "Wykryto twarzy: {0}".format(len(faces))

            # pokaz
        cv2.imshow('Wykrywanie twarzy:', img)
        cv2.imwrite('data/detected.jpg', frame);0

        if cv2.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
     

if __name__ == "__main__":
    main()
