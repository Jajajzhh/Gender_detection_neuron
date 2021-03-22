# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv


# Utiliser le model creer par train.py
model = load_model("simple_test.model")

# ouvrir webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
classes = ['man','woman']

# loop
while webcam.isOpened():

    # lire chaque frame de webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # application du face detection
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    # On fait une detection de gender pour chaque visage detecté, c'est la même chose que la version Photo
    #dans detect-gender.py vous trouverez plus de details
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        face_crop = np.copy(frame[startY:endY,startX:endX])
        #Il est possible que il n'y a pas de visage detecté dans webcam ou le forme du visage ne conforme pas
        #On ne fait pas la detection pour ces situations
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        # preprocessing pour gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        
        # application du gender detection model
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)

        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # Ecrire le gender et son confidence autour du visage
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Affichage de ce frame
    cv2.imshow("gender detection", frame)

    # la clé 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# finir le program et libérer les ressources
webcam.release()
cv2.destroyAllWindows()
