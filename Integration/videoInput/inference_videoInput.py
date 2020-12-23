from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
import keras
import matplotlib.pyplot as plt
import sys
from keras.preprocessing import image
import numpy as np
import glob
import os
import cv2

HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
class_list = ["arterial","connector","local","sub_arterial"]
FC_LAYERS = [256, 128]
dropout = 0.5

#weight_file = sys.argv[1]

def showImage(testImageRaw, label):
	
	
	plt.imshow(testImageRaw)
	plt.title(label)
	plt.show()
	
	'''
	cv2.namedWindow(label, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(label, 600,600)
	cv2.imshow(label, testImageRaw)
	cv2.waitKey(0)
	'''
	return 0
	

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = Flatten()(x)

	for fc in fc_layers:
		x = Dense(fc, activation='relu')(x)
		x = Dropout(dropout)(x)

	preditions = Dense(num_classes, activation='softmax')(x)
	finetune_model = Model(inputs = base_model.input, outputs = preditions)

	return finetune_model


base_model = ResNet50(weights = 'imagenet',
					   include_top = False,
					   input_shape = (HEIGHT, WIDTH, 3))

finetune_model = build_finetune_model(base_model,
									  dropout = dropout,
									  fc_layers = FC_LAYERS,
									  num_classes = len(class_list))

pwd = os.getcwd()

# Please use "pwd+weight_file" if the weight file is saved in the root directory of the project
finetune_model.load_weights("/home/kishor/GWM/DataSets/US_Scenario_Video/checkpointsResNet50_model_weights_Apr_3_e30.h5")

'''
predict_image_path = pwd+"/test_images/"
os.getcwd()
os.chdir(predict_image_path)

images = []
images = glob.glob('*')
'''

# Video to Frame

pwd = os.getcwd()
videoDir = "/home/kishor/GWM/DataSets/US_Scenario_Video/sampleVideo/"
os.getcwd()
os.chdir(videoDir)

videos = []
videos = sorted(glob.glob("*.AVI"))
print(len(videos))

for video in videos:
	print(video)
	vidcap = cv2.VideoCapture(video)
	success, testImage = vidcap.read()
	count = 0
	
	while success:
		
		#fileName = videoDir+"frame%d.jpg" % count
		fileName = videoDir+"frame.jpg"
		
		print(fileName)
		cv2.imwrite(fileName, testImage)
		
		#testImageRaw = cv2.imread(fileName)
		#testImageRaw = cv2.resize(testImageRaw, (300, 300))
		
		testImageRaw = cv2.imread(fileName)
		y = 800
		x = 960
		h = 0
		w = 900
		crop_img = testImageRaw[h:y, x-w:x+w]
		#cv2.imwrite(fileName, crop_img)
		#testImageRaw = cv2.imread(fileName)
		testImageRaw = cv2.resize(crop_img, (300, 300))
		
		testImage = image.img_to_array(testImageRaw)
		testImage = np.expand_dims(testImage, axis = 0)
		
		result = finetune_model.predict(testImage)
		
		result = result.argmax(axis=1)[0]
		label = class_list[result]	
		
		#cv2.imshow(label, np.array(testImageRaw))
		#cv2.waitKey(1)
		
		# Text inside the image
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,250)
		fontScale              = 1
		fontColor              = (0,0,255)
		lineType               = 2


		img = np.array(testImageRaw)
		cv2.putText(img,label, 
    			    bottomLeftCornerOfText, 
    			    font, 
    			    fontScale,
    			    fontColor,
    			    lineType)
		
		cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Window", 600,600)
		cv2.imshow("Window", img)
		cv2.waitKey(1)
		
		print(label)
		
		success, testImage = vidcap.read()
		count += 1

	os.chdir(videoDir)


