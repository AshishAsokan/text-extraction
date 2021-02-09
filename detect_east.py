import pytesseract as pyt
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


# Reading the input image
input_image = cv2.imread('sample.png')
# input_image = cv2.resize(input_image, (640, 320), interpolation=cv2.INTER_AREA)
final_image = input_image.copy()
height, width, _ = input_image.shape

# Reading the new height and width of the image
h_new = 320
w_new = 320
thresh = 0.5

# Choosing dimensions which are multiples of 32
ratio_h = height / float(h_new)
ratio_w = width / float(w_new)

# Resizing the image
image_res = cv2.resize(input_image, (w_new, h_new))
height, width, _ = image_res.shape

# Selecting the layers from the EAST model
layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

# Loading the EAST detector
model = cv2.dnn.readNet('east_model.pb')

blob = cv2.dnn.blobFromImage(image_res, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# Forward Propagation of the model to get outputs
model.setInput(blob)
(scores, geometry) = model.forward(layers)

(n_r, n_c) = scores.shape[2:4]
rectangles = []
conf = []

print('Preprocessing Done')

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
        if prob[j] < thresh:
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

# Applying non-max suppression to avoid overlapping boxes
boxes = non_max_suppression(np.array(rectangles), probs = prob)

# Drawing bounding boxes on the image
for box in boxes:

    # Calculating coordinates for original image
    x1 = int(box[0] * ratio_w)
    y1 = int(box[1] * ratio_h)
    x2 = int(box[2] * ratio_w)
    y2 = int(box[3] * ratio_h)

    # Extracting the text portion of the image
    text_image = np.array(final_image[y1 : y2, x1 : x2])
    text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    detected_text = pyt.image_to_string(text_image)

    # Drawing bounding box on the image
    print('Detected Text:', detected_text)
    cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# Display result
cv2.imshow('Frame', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


