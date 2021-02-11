import pytesseract as pyt
from imutils.object_detection import non_max_suppression
import numpy as np
import sys
import cv2

class TextDetectEast:

    def __init__(self, h_new, w_new, thresh):

        self.h_new = h_new
        self.w_new = w_new
        self.thresh = thresh

        # Layers of the EAST model
        self.layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    def preprocess(self, input_image):

        final_image = input_image.copy()
        h, w, _ = input_image.shape

        # Calculating ratios
        self.ratio_h = h / float(self.h_new)
        self.ratio_w = w / float(self.w_new)

        # Resizing the image
        image_res = cv2.resize(input_image, (self.w_new, self.h_new))
        self.h, self.w, _ = image_res.shape
        self.image = final_image

        # Loading the model and generating blob from image
        model = cv2.dnn.readNet('src/east_model.pb')
        blob = cv2.dnn.blobFromImage(image_res, 1.0, (self.w, self.h), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Forward Propagation of the model
        model.setInput(blob)
        (scores, geometry) = model.forward(self.layers)
        self.scores = scores
        self.geometry = geometry

# Forward Propagation of the model to get outputs

    def parse_output(self):

        # Using class instances
        (n_r, n_c) = self.scores.shape[2:4]
        scores = self.scores
        geometry = self.geometry

        rectangles = []
        conf = []

        # Iterating over the number of rows 
        for i in range(0, n_r):

            prob = scores[0, 0, i]

            # Geometrical data
            x_1 = geometry[0, 0, i]
            x_2 = geometry[0, 1, i]
            x_3 = geometry[0, 2, i]
            x_4 = geometry[0, 3, i]

            # Angle data
            angles = geometry[0, 4, i]

            # Going over column values
            for j in range(0, n_c):

                # If score < threshold
                if prob[j] < self.thresh:
                    continue

                # Scaling the factors
                offset_x = j * 4.0
                offset_y = i * 4.0

                # Computing sin, cos, width and height
                cos = np.cos(angles[j])
                sin = np.sin(angles[j])
                h = x_1[j] + x_3[j]
                w = x_2[j] + x_4[j]

                # Computing bounding box coordinates
                x2 = int(offset_x + (cos * x_2[j]) + sin * x_3[j])
                y2 = int(offset_y + (cos * x_3[j]) - sin * x_2[j])
                x1 = int(x2 - w)
                y1 = int(y2 - h)

                # Bounding box details
                rectangles.append((x1, y1, x2, y2))
                conf.append(prob[j])

        self.rectangles = rectangles
        self.conf = conf

    def draw_box(self):

        # Applying non-max suppression to avoid overlapping boxes
        boxes = non_max_suppression(np.array(self.rectangles), probs = self.conf)

        # Drawing bounding boxes on the image
        for box in boxes:

            # Calculating coordinates for original image
            x1 = int(box[0] * self.ratio_w)
            y1 = int(box[1] * self.ratio_h)
            x2 = int(box[2] * self.ratio_w)
            y2 = int(box[3] * self.ratio_h)

            # Extracting the text portion of the image
            text_image = np.array(self.image[y1 : y2, x1 : x2])

            # Gives exception for some frames
            try:
                text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
            except:
                continue
            self.detected_text += pyt.image_to_string(text_image)

            # Drawing bounding box on the image
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    def text_detect(self, image):
        self.detected_text = ""
        self.preprocess(image)
        self.parse_output()
        self.draw_box()

        return self.image, self.detected_text


if __name__ == '__main__':

    # Creating model and reading image
    east_obj = TextDetectEast(320, 320, 0.5)
    input_image = cv2.imread(sys.argv[1])

    # Using EAST to detect text
    image, text = east_obj.text_detect(input_image)
    print('Text:', text)
    
    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


