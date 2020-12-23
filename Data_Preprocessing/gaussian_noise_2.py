# Adding Gaussian noise using skimage - random_noise

from PIL import Image
import numpy as np
from skimage.util import random_noise
import sys

noise_type = sys.argv[1]

filename = "test.png"
im = Image.open(filename)
# convert PIL Image to ndarray
im_arr = np.asarray(im)
# random_noise() method will convert image in [0, 255] to [0, 1.0],
# inherently it use np.random.normal() to create normal distribution
# and adds the generated noised back to image

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
img.show()
img.save(noise_type+"_"+filename, format='PNG')

