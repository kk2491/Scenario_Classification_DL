from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
import numpy
from sklearn.metrics import classification_report, confusion_matrix
import keras
import matplotlib.pyplot as plt
import pandas as pd
import os

HEIGHT = 300
WIDTH = 300
TRAIN_DIR = "dataset/Training_set/"
TEST_DIR = "dataset/Test_set/"
TENSOR_DIR = "Tensorlog/"
BATCH_SIZE = 8
class_list = ["booth", "highway", "open_non_highway", "overpass", "settlement", "traffic_road", "tunnel", "tunnel_exit"]
FC_LAYERS = [256, 128]
dropout = 0.5
NUM_EPOCHS = 30

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = Flatten()(x)

	for fc in fc_layers:
		print(fc)
		x = Dense(fc, activation='relu')(x)
		x = Dropout(dropout)(x)

	preditions = Dense(num_classes, activation='softmax')(x)
	finetune_model = Model(inputs = base_model.input, outputs = preditions)

	return finetune_model


base_model = ResNet50(weights = 'imagenet',
					   include_top = False,
					   input_shape = (HEIGHT, WIDTH, 3))

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
								   horizontal_flip = True,
								   vertical_flip = False)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
								  horizontal_flip = True,
								  vertical_flip = False)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
													target_size = (HEIGHT, WIDTH),
													batch_size = BATCH_SIZE,
													shuffle = False)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
												  target_size = (HEIGHT, WIDTH),
												  batch_size = BATCH_SIZE,
												  shuffle = False)

finetune_model = build_finetune_model(base_model,
									  dropout = dropout,
									  fc_layers = FC_LAYERS,
									  num_classes = len(class_list))

adamOptimizer = Adam(lr = 0.00001)
finetune_model.compile(adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy"])

filepath = "./checkpoints" + "RestNet50" + "_model_weights_Apr_3_e30.h5"
cb_weight = keras.callbacks.ModelCheckpoint(filepath, monitor = ["acc"], verbose= 1, mode = "max")
# callBack=TensorBoard(log_dir=("/home/ubuntu/GWM/Scenario_Classification/"))
cb_tensor = TensorBoard(log_dir=TENSOR_DIR)
callbacks_list = [cb_weight, cb_tensor]

print(train_generator.class_indices)

weights = finetune_model.fit_generator(generator = train_generator, epochs = NUM_EPOCHS, steps_per_epoch = train_generator.samples // BATCH_SIZE, 
									   shuffle = True, callbacks=callbacks_list, validation_data = test_generator)

print("Number of training samples : {}".format(train_generator.samples))
test_steps_for_epoch = numpy.math.ceil(train_generator.samples/train_generator.batch_size)
predictions=finetune_model.predict_generator(train_generator,steps=test_steps_for_epoch)
predicted_classes=numpy.argmax(predictions,axis=1)
true_classes=train_generator.classes
class_labels=list(train_generator.class_indices.keys())
print(class_labels)
report=classification_report(true_classes,predicted_classes,target_names=class_labels)
print("Report - Training Dataset")
print(report)
confusionMatrix = confusion_matrix(true_classes,predicted_classes)
print("Confusion Matrix - Training Dataset")
#print(confusionMatrix)
print(pd.DataFrame(confusionMatrix, index = class_labels, columns = class_labels))


print("Number of test samples : {}".format(test_generator.samples))
test_steps_for_epoch = numpy.math.ceil(test_generator.samples/test_generator.batch_size)
predictions=finetune_model.predict_generator(test_generator,steps=test_steps_for_epoch)
predicted_classes=numpy.argmax(predictions,axis=1)
true_classes=test_generator.classes
class_labels=list(test_generator.class_indices.keys())
print(class_labels)
report=classification_report(true_classes,predicted_classes,target_names=class_labels)
print("Report - Testing Dataset")
print(report)
confusionMatrix = confusion_matrix(true_classes,predicted_classes)
print("Confusion Matrix - Testing Dataset")
#print(confusionMatrix)
print(pd.DataFrame(confusionMatrix, index = class_labels, columns = class_labels))
