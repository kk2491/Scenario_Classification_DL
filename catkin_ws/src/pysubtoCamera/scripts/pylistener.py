#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge

# Deep Learning Start
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

#print(tensorflow.__version__)
#print(keras.__version__)


HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
class_list = ["booth", "highway", "open_non_highway", "overpass", "settlement", "traffic_road", "tunnel", "tunnel_exit"]
FC_LAYERS = [256, 128]
dropout = 0.5

# Deep Learning Start

# Build fine tune model
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

# Load the base model and weights
def load_base_model():
    base_model = ResNet50(weights = 'imagenet',
                       include_top = False,
                       input_shape = (HEIGHT, WIDTH, 3))
    print("Base model loaded ..!!!")
    #finetune_model.load_weights("/home/kishor/rostensor_v1/src/pysubtoCamera/scripts/checkpointsRestNet50_model_weights_Apr_3_e30.h5")

    #print("Weight Loaded...!!!")

    return base_model

base_model = load_base_model()

finetune_model = build_finetune_model(base_model,
                                      dropout = dropout,
                                      fc_layers = FC_LAYERS,
                                      num_classes = len(class_list))


finetune_model.load_weights("/home/Ubuntu_1/ROS/rostensor_v1/src/pysubtoCamera/scripts/checkpointsRestNet50_model_weights_Apr_3_e30.h5")
finetune_model._make_predict_function()

pub = rospy.Publisher('chatter', String, queue_size=10)

def predictScenario(mymodel):

    test_image = image.load_img(path = "/home/kishor/rostensor_v1/src/pysubtoCamera/scripts/000030.png", target_size = (300, 300))
    test_image = image.img_to_array(test_image)
    #test_image = np.resize(test_image, (300, 300))
    test_image = np.expand_dims(test_image, axis = 0)

    result = mymodel.predict(test_image)
    print(result)
    result = result.argmax(axis=1)[0]
    label = class_list[result]
    print("Image : Index : {} || Label : {}".format(result, label))

    
def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    #print("I heard something")
    bridge = CvBridge()
    test_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 600,600)
    #cv2.imshow("image", test_image)
    #cv2.waitKey(1)    

    cv2.imwrite("File.jpg", test_image)

    test_image = image.load_img(path = "/home/Ubuntu_1/ROS/rostensor_v1/File.jpg", target_size = (300, 300))
    test_image = image.img_to_array(test_image)
    
    
    #test_image = np.array(test_image, dtype=np.uint8)
    #test_image = image.img_to_array(cv_image)
    #test_image = np.resize(test_image, (300, 300))
    #print("Shape Before : {}".format(test_image.shape))
    test_image = np.expand_dims(test_image, axis = 0)
    #print("Shape After  : {}".format(test_image.shape))
    result = finetune_model.predict(test_image)
    #print(result)
    result = result.argmax(axis=1)[0]
    label = class_list[result]
    print("Image : Index : {} || Label : {}".format(result, label))
    
    pub.publish(label)

    
def listener():

    print("Node initialization START..!!!")

    rospy.init_node('pycameralistener', anonymous=True)

    rospy.Subscriber('/apollo/sensor/camera/traffic/image_long', Image, callback)

    print("Node initialization END and SPIN START")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == "__main__":
    listener()
