# http://scikit-image.org/docs/dev/api/skimage.util.html
# https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv

import glob
from PIL import Image
import numpy as np
from skimage.util import random_noise
import sys
import os

noise_type = sys.argv[2]
class_type = sys.argv[1]
# dest_folder = sys.argv[2]

fm2 = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/rotated_images/"
source_dir = fm2+"rotated_"+class_type+"/"
print(source_dir)
dest_dir = fm2+noise_type+"_"+class_type

print(os.getcwd())
os.chdir(source_dir)
print(os.getcwd())

images = sorted(glob.glob("*.png"))
print(images)

for image in images:
	im = Image.open(source_dir+image)
	im_arr = np.asarray(im)
	if (noise_type == "gaussian"):
		noise_img = random_noise(im_arr, mode='gaussian', var=0.05**2)
	elif (noise_type == "localvar"):
		noise_img = random_noise(im_arr, mode='localvar')
	elif (noise_type == "poisson"):
		noise_img = random_noise(im_arr, mode='poisson')
	else:
		print("No distribution specified")

	noise_img = (255*noise_img).astype(np.uint8)
	img = Image.fromarray(noise_img)
	#img.show()
	os.chdir(dest_dir)
	img.save(noise_type+"_"+image, format='PNG')


