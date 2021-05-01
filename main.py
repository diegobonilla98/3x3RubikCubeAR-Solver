import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from utils import process_image

cam = cv2.VideoCapture('MAH03111.MP4')
last_face = None
colors_encoded = ["W", "Y", "B", "G", "O", "R"]
face_colors = ["w", "#ffcf00",
               "#00008f", "#009f0f",
               "#ff6f00", "#cf0000"]
original_colors = np.load('original_colors.npy')
converted_colors = original_colors.copy()
colors_decoder = dict(zip(colors_encoded, face_colors))
while True:
    ret, frame = cam.read()
    if not ret:
        np.save('colors.npy', converted_colors)
        break
    frame = cv2.resize(frame, (5456 // 4, 3064 // 4))

    canvas, face_colors = process_image(frame)
    if face_colors is not None:
        if last_face is None:
            last_face = face_colors
            center = face_colors[1][1]
            idx = np.where(original_colors == colors_decoder[center])
            converted_colors[idx] = [colors_decoder[c] for c in face_colors.flatten()]
        elif not np.array_equal(face_colors, last_face):
            last_face = face_colors
            center = face_colors[1][1]
            idx = np.where(original_colors == colors_decoder[center])
            converted_colors[idx] = [colors_decoder[c] for c in face_colors.flatten()]
    cv2.imshow("Result", canvas)

    key = cv2.waitKey(1)
    if key == ord(' '):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, ::-1]
        plt.imshow(hsv)
        plt.show()
    if key == ord('q'):
        break
