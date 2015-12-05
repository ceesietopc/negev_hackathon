#!/usr/bin/env python

import os
from PIL import Image
import numpy as np
import cv2

recognizer = cv2.createLBPHFaceRecognizer()

cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

def get_images_and_labels(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.endswith('.sad') and not f.endswith('.DS_Store')]

    images = []
    labels = []

    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')

        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("zdjecie", ""))
        faces = faceCascade.detectMultiScale(image)

        for (x, y, w, h) in faces:
            images.append(image[y:y+h, x:x+w])
            labels.append(nbr)
            cv2.imshow("Dodawanie zdjec do data_samples", image[y:y+h, x:x+w])
            cv2.waitKey(50)

    return images, labels

path = "data"
path2 = "data_samples"

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))

image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.karate')]

for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)

    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y:y+h, x:x+w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("zdjecie", ""))

        if nbr_actual == nbr_predicted:
            print "{} poprawnie rozpoznano z pewnoscia {} ".format(nbr_actual, conf)
        else:
            print "{} zle rozpoznano jako {} ".format(nbr_actual, nbr_predicted)
        cv2.imshow("Rozpoznawanie twarzy", predict_image[y:y+h, x:x+w])
        cv2.waitKey(1000)

