import os
import glob

pwd = os.getcwd()
folderDir = pwd+"/"+"Images/"
os.chdir(folderDir)
subDirs = sorted(glob.glob("*"))
print(subDirs)

for subDir in subDirs:
	
	pwd = os.getcwd()
	path = pwd+"/"+subDir+"/"
	print(path)
	os.chdir(path)

	imageLists = sorted(glob.glob("*"))

	for imageName in imageLists:
		new_name = subDir+"_"+imageName
		os.rename(imageName, new_name)

	os.chdir(folderDir)
