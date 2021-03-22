

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv
import glob
# Lire les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()
# detect_gender.py -i example.png
#Definir le repertoire et le nom du model deja entrainee
model_path = "simple_test.model"
# Lire l'image par le repertoire donne en -i
print(args.image)
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# Utiliser le model creer par train.py
model = load_model(model_path)

# detection de visage a l'aide de cvlib.detect_face function
face, confidence = cv.detect_face(image)

classes = ['man','woman']

# Il exsit possiblement plusieur visages detecté donc on fait loop par chaque un
for idx, f in enumerate(face):

     # trouver les points de coin pour le rectangle
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

    # tracer le rectangle autour de visage detecter
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    #creer la partie du visage découpé
    face_crop = np.copy(image[startY:endY,startX:endX])

    # preprocessing pour gender detection model
    try:
        face_crop = cv2.resize(face_crop, (96,96))
    except Exception as e:
        print(str(e))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    #L'entree de donnes est de 4 dimension donc on ajout une dimension
    face_crop = np.expand_dims(face_crop, axis=0)

    # application du gender detection model
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # Prend le gender avec la confidence le plus haute
    idx = np.argmax(conf)
    label = classes[idx]
    
    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # Ecrire le gender et son confidence autour du visage
    cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

# Afficher l'image avec detection
cv2.imshow("gender detection", image)
cv2.waitKey()

# Enregistrer l'image
cv2.imwrite("gender_detection.jpg", image)

cv2.destroyAllWindows()
