from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model 
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
import keras
import matplotlib.pyplot as plt

HEIGHT = 300
WIDTH = 300
TRAIN_DIR = "/home/kishor/GWM/Gitlab_Repos/scenario_classification_deep_learning/dataset/training_set/"
#TEST_DIR = "/home/ubuntu/GWM/Carla_V3/dataset/test_set/"
BATCH_SIZE = 8
class_list = ["booth", "highway", "open_non_highway", "overpass", "settlement", "traffic_road", "tunnel", "tunnel_exit"]
FC_LAYERS = [100, 20]
dropout = 0.5
NUM_EPOCHS = 100
BATCH_SIZE = 8
#num_train_images = 100

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


def plot_training(history):
	acc = history.history["acc"]
	val_acc = history.history["val_acc"]
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	epochs = range(len(acc))

	plt.plot(epochs, acc, 'r')
	plt.plot(epochs, val_acc, 'r')
	plt.title("Training and validation accuracy")

	plt.show()
	plt.savefig('acc_vs_epochs.png')

base_model = ResNet50(weights = 'imagenet',
					   include_top = False,
					   input_shape = (HEIGHT, WIDTH, 3))

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
								   rotation_range = 90,
								   horizontal_flip = True,
								   vertical_flip = False)

#test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
#								  rotation_range = 90,
#								  horizontal_flip = True,
#								  vertical_flip = False)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
													target_size = (HEIGHT, WIDTH),
													batch_size = BATCH_SIZE)

#test_generator = test_datagen.flow_from_directory(TEST_DIR,
#												  target_size = (HEIGHT, WIDTH),
#												  batch_size = BATCH_SIZE)

finetune_model = build_finetune_model(base_model,
									  dropout = dropout,
									  fc_layers = FC_LAYERS,
									  num_classes = len(class_list))

adam = Adam(lr = 0.00001)
finetune_model.compile(adam, loss="categorical_crossentropy", metrics=["accuracy"])

filepath = "./checkpoints" + "RestNet50" + "_model_weights_March_13_e100.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = ["acc"], verbose= 1, mode = "max")
cb=TensorBoard(log_dir=("/home/kishor/GWM/Gitlab_Repos/scenario_classification_deep_learning/"))
callbacks_list = [checkpoint, cb]

# cb=TensorBoard(log_dir=("/home/ubuntu/GWM/Carla_V3/"))

print(train_generator.class_indices)

history = finetune_model.fit_generator(generator = train_generator, epochs = NUM_EPOCHS, steps_per_epoch = 100, 
									   shuffle = True, callbacks=callbacks_list)

plot_training(history)


