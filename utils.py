import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


red_range = [(170, 220, 170), (240, 255, 185)]
blue_range = [(180, 250, 100), (250, 255, 120)]
green_range = [(130, 200, 60), (225, 255, 73)]
white_range = [(200, 20, 100), (255, 45, 115)]
orange_range = [(200, 230, 5), (255, 255, 17)]
yellow_range = [(220, 185, 25), (255, 255, 33)]

COLOR_NAMES = ["R", "B", "G", "W", "O", "Y"]
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 100, 255), (0, 255, 255)]


def process_image(img):
    canvas = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, ::-1]
    total_mask = np.zeros(img.shape[:2], np.uint8)
    for rang, col in zip([red_range, blue_range, green_range, white_range, orange_range, yellow_range], COLOR_NAMES):
        mask = cv2.inRange(cv2.blur(hsv, (10, 10)), rang[0], rang[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
        total_mask = cv2.bitwise_or(total_mask, mask)
    cube_mask = cv2.morphologyEx(total_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
                                 iterations=2)
    contours, _ = cv2.findContours(cube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return canvas, None
    cube_contour = max(contours, key=lambda x: cv2.contourArea(x))
    cube_x, cube_y, cube_w, cube_h = cv2.boundingRect(cube_contour)

    if abs(cube_w - cube_h) > 10 or cube_w < 250:
        return canvas, None

    rect_size = int(((cube_w + cube_h) / 2) / 3)
    colors_face = np.zeros((3, 3), np.str)
    for i in range(1, 4):
        for j in range(1, 4):
            xx = int(np.clip(cube_x - rect_size // 2 + rect_size * i, 0, img.shape[1] - 1))
            yy = int(np.clip(cube_y - rect_size // 2 + rect_size * j, 0, img.shape[0] - 1))
            color = hsv[yy, xx]
            c = None
            for rang, col, cols in zip([red_range, blue_range, green_range, white_range, orange_range, yellow_range], COLOR_NAMES, COLORS):
                color_mask = cv2.inRange(color.reshape(1, 1, 3), rang[0], rang[1])
                if color_mask[0][0] == 255:
                    colors_face[i-1, j-1] = col
                    c = col
                    break
            if c is None:
                return canvas, None
            cv2.circle(canvas, (xx, yy), 10, 0, 2)
            cv2.putText(canvas, c, (xx + 5, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 0), 2)
    cv2.rectangle(canvas, (cube_x, cube_y), (cube_x + cube_w, cube_y + cube_h), (0, 255, 0), 2)
    return canvas, colors_face.T
