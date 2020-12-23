import pandas
import glob
import os
import shutil

classes = ["highway", "open_non_highway", "overpass", "settlement", "traffic_road", "tunnel", "tunnel_exit", "booth"]

pwd = os.getcwd()
label_file =  pandas.read_csv(pwd+"/class_labels.csv", sep = " ", header = None)
print(pwd)

for folder in classes:
	os.mkdir(pwd+"/"+folder)

highway = pwd+"/highway/"
open_non_highway = pwd+"/open_non_highway/"
overpass = pwd+"/overpass/"
settlement = pwd+"/settlement/"
traffic_road = pwd+"/traffic_road/"
tunnel = pwd+"/tunnel/"
tunnel_exit = pwd+"/tunnel_exit/"
booth = pwd+"/booth/"


num_rows = label_file.shape[0]
print(num_rows)
print(label_file.head())

os.chdir(pwd+"/img/")
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
		print(highway)
	elif label == 1:
		#copy to open_highway
		shutil.copy(image, open_non_highway)
		print(open_non_highway)
	elif label == 2:
		#copy to tunnel
		shutil.copy(image, tunnel)
		print(tunnel)
	elif label == 3:
		#copy to tunnel_exit
		shutil.copy(image, tunnel_exit)
		print(tunnel_exit)
	elif label == 4:
		#copy to settlement
		shutil.copy(image, settlement)
		print(settlement)
	elif label == 5:
		#copy to overpass
		shutil.copy(image, overpass)
		print(overpass)
	elif label == 6:
		#copy to booth
		shutil.copy(image, booth)
		print(booth)
	elif label == 7:
		#copy to traffic_road
		shutil.copy(image, traffic_road)
		print(traffic_road)
	else:
		pass

print("File copy completed")	

