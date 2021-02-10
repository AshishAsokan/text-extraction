import cv2
import sys
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from imutils.object_detection import non_max_suppression
import pytesseract as pyt
from scipy import ndimage
import numpy as np

from detect_east import TextDetectEast
from detect_watershed import LineSegment

def video_text_detect(video_path, text_model):

    video_obj = cv2.VideoCapture(video_path)
    frame_count = 0
    while video_obj.isOpened():

        ret, frame = video_obj.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 100 != 0:
            continue

        # Detecting the text using the model
        image, text = text_model.text_detect(frame)
        if len(text) > 1:
            print('Frame Number:', frame_count)
            print(" Text:", text)
            print('*' * 50)

        # Displaying the frame
        cv2.imshow('Video Frame', image)
        key = cv2.waitKey(0)

        # Exit on escape
        if key == 27:
            break

    video_obj.release()


if __name__ == '__main__':

    # Creating the objects
    text_model = None

    # 1 for watershed model
    if sys.argv[1] == '1':
        text_model = LineSegment()

    # 2 for EAST model
    elif sys.argv[1] == '2':
        text_model = TextDetectEast(320, 320, 0.5)

    # Invalid input
    else:
        print('Invalid Command!')
        exit()

    video_text_detect(sys.argv[2], text_model)

    