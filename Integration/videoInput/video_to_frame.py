import cv2
import glob
import os

pwd = os.getcwd()

videoDir = "/home/kishor/GWM/DataSets/US_Scenario_Video/Videos/"
os.getcwd()
os.chdir(videoDir)

videos = []
videos = sorted(glob.glob("*.AVI"))
print(len(videos))

for video in videos:
	print(video)
	vidcap = cv2.VideoCapture(video)
	success, image = vidcap.read()
	count = 0

	# Create directory for each video
	os.chdir(pwd)
	os.mkdir(pwd+"/"+video)

	while success:
		cv2.imwrite(pwd+"/"+video+"/"+"frame%d.jpg" % count, image)
		success, image = vidcap.read()
		print("Read a new frame : ", success)
		count += 1

	os.chdir(videoDir)
'''
vidcap = cv2.VideoCapture('arterial_suburban1.AVI')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
'''
