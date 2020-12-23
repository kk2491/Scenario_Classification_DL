import pandas
import glob
import os
import shutil

highway = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/highway/"
open_highway = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/open_non_highway"
overpass = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/overpass/"
settlement = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/settlement/"
traffic_road = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/traffic_road/"
tunnel = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/tunnel/"
tunnel_exit = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/tunnel_exit/"
booth = "/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/booth/"

label_file = pandas.read_csv("/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/class_labels.csv", sep = " ", header = None)
num_rows = label_file.shape[0]
print(num_rows)
print(label_file.head())

os.chdir("/home/kishor/GWM/DataSets/FM2/unizg-fer-fm2_single_label/img/")
print(os.getcwd())

image_files = sorted(glob.glob("*.png"))
#print(image_files)

for index, image in enumerate(image_files):
	#print(image)
	#print(label_file.loc[index])
	
	for col in range(8):
		#print(label_file[col].loc[index])
		if (label_file[col].loc[index] == 1):
			# Get the index of column
			label = col
			break
	
	print("Image : {}   Label : {}".format(image, label))
	
	if label == 0:
		#copy to highway
		shutil.copy(image, highway)
	elif label == 1:
		#copy to open_highway
		shutil.copy(image, open_highway)
	elif label == 2:
		#copy to tunnel
		shutil.copy(image, tunnel)
	elif label == 3:
		#copy to tunnel_exit
		shutil.copy(image, tunnel_exit)
	elif label == 4:
		#copy to settlement
		shutil.copy(image, settlement)
	elif label == 5:
		#copy to overpass
		shutil.copy(image, overpass)
	elif label == 6:
		#copy to booth
		shutil.copy(image, booth)
	elif label == 7:
		#copy to traffic_road
		shutil.copy(image, traffic_road)
	else:
		pass

print("File copy completed")	
			
