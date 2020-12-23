import cv2
import numpy as np 
import glob
import os

pwd = os.getcwd()
folderDir = pwd+"/"+"Images/"
os.chdir(folderDir)
subDirs = sorted(glob.glob("*"))
print(subDirs)

for subDir in subDirs:
	
	pwd = os.getcwd()
	path = pwd+"/"+subDir+"/"
	print(path)
	os.chdir(path)

	imageLists = sorted(glob.glob("*"))

	for imageName in imageLists:
		image = cv2.imread(imageName)
		y = 800
		x = 960
		h = 0
		w = 900
		crop_img = image[h:y, x-w:x+w]
		cv2.imwrite(imageName, crop_img)

		#cv2.namedWindow('window1', cv2.WINDOW_NORMAL)
		#cv2.resizeWindow("window1", 600, 600)
		#cv2.imshow("window1", image)
		#cv2.waitKey(-1)
		print(image.shape)

	os.chdir(folderDir)

'''
cv2.namedWindow('window1', cv2.WINDOW_NORMAL)
cv2.resizeWindow("window1", 600, 600)
cv2.imshow("window1", image)
#cv2.waitKey(-1)
print(image.shape)

# image size - 1080 x 1920

y = 800
x = 960
h = 0
w = 900
crop_img = image[h:y, x-w:x+w]

cv2.namedWindow('window2', cv2.WINDOW_NORMAL)
cv2.resizeWindow("window2", 600, 600)
cv2.imshow("window2", crop_img)
cv2.waitKey(-1)
'''
