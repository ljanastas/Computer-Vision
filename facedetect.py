# Detect faces using haarcascades in python

import cv2
import numpy as np

def detect(img, cascade_fn='haarcascade_frontalface_default.xml',
                   scaleFactor=1.1, minNeighbors=3):
        cascade = cv2.CascadeClassifier(cascade_fn)
        rects = cascade.detectMultiScale(img, scaleFactor=scaleFactor,minNeighbors=minNeighbors)
        return rects

def largest_face(faces):
    biggest_area = 0
    biggest_face = 0
    for (x,y,w,h) in faces:
        if w*h > biggest_area:
            biggest_area = w*h
            biggest_face = [x,y,w,h]
    return biggest_face

files = np.loadtxt('filenames.txt', dtype=str)

zero_faces = []
many_faces = {}
one_face = {}
i = 0
for f in files:
    i += 1
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect(gray)
    if len(faces) == 0:
        zero_faces.append(f)
    elif len(faces) > 1:
        many_faces[f] = largest_face(faces)
    else:
        one_face[f] = faces[0]

# Fix images with no detections
fix_faces = {}
for f in zero_faces:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect(gray, scaleFactor=1.04, minNeighbors=1)
    alt_faces = detect(gray, scaleFactor=1.07, minNeighbors=1)
    if len(alt_faces) > 0:
        if len(faces) > 0:
            faces = append(faces, alt_faces, axis=0)
        else:
            faces = alt_faces
    fix_faces[f] = largest_face(faces)

# Make list of faces corresponding to the files list
# Remove files that don't have a corresponding face
all_faces = []
all_files = []
for l in files:
    if l in one_face.keys():
        all_files.append(l + '\n')
        all_faces.append(one_face[l])
    elif l in many_faces.keys():
        all_files.append(l + '\n')
        all_faces.append(many_faces[l])
    elif l in fix_faces.keys():
        all_files.append(l + '\n')
        all_faces.append(fix_faces[l])

with open('files.txt', 'w') as f:
    f.writelines(all_files)

np.savetxt('faces.txt', all_faces)
