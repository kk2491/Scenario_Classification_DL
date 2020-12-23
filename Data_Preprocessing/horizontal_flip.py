# Horizontal Flip
import cv2
import numpy as np
file_name = "test.png"
img = cv2.imread(file_name)
flipped = np.fliplr(img)
cv2.imshow("flipped", flipped)
cv2.waitKey(0)
cv2.imwrite("flip_"+file_name, flipped)