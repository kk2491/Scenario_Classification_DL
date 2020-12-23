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

HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
class_list = ["booth", "highway", "open_non_highway", "overpass", "settlement", "traffic_road", "tunnel", "tunnel_exit"]
FC_LAYERS = [256, 128]
dropout = 0.5

weight_file = sys.argv[1]

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
finetune_model.load_weights(pwd+"/Weights/"+weight_file)
predict_image_path = pwd+"/dataset/raw_image/"

os.getcwd()
os.chdir(predict_image_path)

images = []
images = glob.glob('*')

for imagepath in images:

	test_image = image.load_img(path = predict_image_path+'/'+imagepath, target_size = (300, 300))
	plt.imshow(test_image)
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)

	#print(test_image.shape())
	result = finetune_model.predict(test_image)
	print(result)
	result = result.argmax(axis=1)[0]
	label = class_list[result]
	plt.title(label)
	plt.show()
	print("Image : {} || Index : {} || Label : {}".format(imagepath, result, label))
