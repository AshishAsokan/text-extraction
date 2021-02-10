# Importing all necessary modules
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import pytesseract as pyt
from scipy import ndimage
import numpy as np


class LineSegment:

    def __init__(self):

        # Initialising Kernels to be used in the program
        self.kernel = np.ones((3, 3), np.uint8)
        self.kernel1 = np.ones((5, 600), np.uint8)

    def preprocess(self, path):

        # Reading image
        img = cv2.imread(path)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, channels = img.shape

        # Thresholding, dilation and erosion
        _, thresh = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img3 = cv2.dilate(thresh, self.kernel1, iterations=1)
        img3 = cv2.erode(img3, np.ones((5, int(height/200)*100), np.uint8), iterations = 2)

        self.image = img
        self.img = img3
        self.img2 = img2

    def connectComponents(self):
        
        _, marked = cv2.connectedComponents(self.img, connectivity=4)
        
        # Mapping each component to corresponding HSV values
        label_hue = np.uint8(179*marked/np.max(marked))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting from HSV to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0] = 0

        # Eroding Image and converting to grayscale
        labeled_img = cv2.erode(labeled_img, self.kernel)
        image = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
        self.img = image

    def Watershed(self):
        
        # Finding Eucledian distance to determine markers
        temp = ndimage.distance_transform_edt(self.img)
        Max = peak_local_max(temp, indices=False, min_distance=30, labels=self.img)
        markers = ndimage.label(Max, structure=np.ones((3, 3)))[0]

        # Applying Watershed to the image and markers
        res = watershed(temp, markers, mask = self.img)

        # If the elements are same, they belong to the same object (Line)
        for i in np.unique(res):
            if i==0:
                continue
            mask = np.zeros(self.img2.shape, dtype="uint8")
            mask[res == i] = 255

            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)

            # Drawing a rectangle around the object
            [x, y, w, h] = cv2.boundingRect(c)
            if((w * h) > 1000):

                # Extracting the text portion of the image
                text_image = np.array(self.image[y : y + h, x : x + w])
                text_image = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
                self.detected_text += pyt.image_to_string(text_image)

                cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    def text_detect(self, path):

        self.detected_text = ""
        self.preprocess(path)
        self.connectComponents()
        self.Watershed()

        return self.image, self.detected_text

if __name__ == '__main__':
    sample = LineSegment()
    image, text = sample.text_detect('images.png')
    print(text)

    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()