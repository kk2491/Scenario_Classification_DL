# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

import numpy as np
import argparse
import imutils
import cv2
import glob
from PIL import Image
import sys
import os

'''
# loop over the rotation angles
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate(image, angle)
	cv2.imshow("Rotated (Problematic)", rotated)
	cv2.waitKey(0)
 
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)
'''

class_type = sys.argv[1]

# dest_folder = sys.argv[2]

fm2 = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/gaussian_images/"
source_dir = fm2+"gaussian_"+class_type+"/"
print(source_dir)
dest_dir = fm2+"rotated"+"_"+class_type+"/"

print(os.getcwd())
os.chdir(source_dir)
print(os.getcwd())

images = sorted(glob.glob("*.png"))
print(images)

for image in images:
	print(image)
	img = cv2.imread(image)
	rotated1 = imutils.rotate(img, 5)
	#cv2.imshow("img", rotated1)
	#cv2.waitKey(0)
	rotated2 = imutils.rotate(img, 355)
	#cv2.imshow("2nd", rotated2)
	#cv2.waitKey(0)
	
	#os.chdir(dest_dir)
	cv2.imwrite(dest_dir+"rotated_1_"+image, rotated1)
	cv2.imwrite(dest_dir+"rotated_2_"+image, rotated2)
	





