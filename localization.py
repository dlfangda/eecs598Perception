import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import csv
import pandas as pd

def  get_proj(folder_str, img_index_str):  #get the projection matrix
	file_name = img_index_str + "_proj.bin"       #img_index_str is in string type,e.g:"0001"
	
	file = glob('./'+folder_str+'/'+file_name)  
	proj = np.fromfile(file[0], dtype = np.float32)   #file is a list of length 1
	proj.resize([3,4])
	return proj

def get_s(folder_str, img_index_str, proj, u, v):  #get the distance from camera to point(u,v)
	file_name = img_index_str + "_cloud.bin"
	file = glob('./'+folder_str+'/'+file_name)
	xyz = np.fromfile(file[0], dtype=np.float32)
	xyz = xyz.reshape([3, -1]) 
	clr = np.linalg.norm(xyz, axis=0) #get the distance from camera to points
	uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
	uv = uv / uv[2, :]
	u_a = np.array(uv[0, :]) - u #to calculate minimum dis
	v_a = np.array(uv[1, :]) - v #to calculate minimum dis
	length = uv[0, :].shape[0] #uv[0,:].shape is in format (int,)
	dis_a = []
	for i in range(length):
		dis = u_a[i]**2 + v_a[i]**2    
		dis_a.append(dis)
	idx = np.argmin(dis_a)     #get the index of the element which is closest to given u,v
	s = clr[idx] #s is the distance from camera to the nearest point to the point(u,v)
	return s

def get_xyz(u, v, s, proj):  #refer to the 3d reconstruction doc shared in google drive
	uvs = np.array([u,v,1]).reshape([3,1])
	uvs = s * uvs
	xyz = np.linalg.pinv(proj) @ uvs
	x = float(xyz[0])
	y = float(xyz[1])
	z = float(xyz[2])+5.5
	return x, y, z

def get_image_size(folder_str, img_index_str):  #get the projection matrix
	file_name = img_index_str + "_image.jpg"       #img_index_str is in string type,e.g:"0001"
	file = glob('./'+folder_str+'/'+file_name) 
	snap = file[0]
	img = plt.imread(snap)
	x = img.shape[0]
	y = img.shape[1]
	return x,y

def write(folder_str, num, guid, u_a, v_a):
	img_index_str = num
	print(img_index_str)
	proj = get_proj(folder_str, img_index_str)
	name = folder_str+'/'+img_index_str
	if name in list(guid):
		print('!!!!!!!!!!!!!!!!')
		n = list(guid).index(name)
		u = u_a[n]
		v = v_a[n]
	else:
		x,y = get_image_size(folder_str, img_index_str)
		u = x/2
		v = y/2
	s = get_s(folder_str, img_index_str, proj, u, v)
	x,y,z = get_xyz(u, v, s, proj)
	file.write(folder_str+'/'+img_index_str+"/x,"+str(x))
	file.write("\n")
	file.write(folder_str+"/"+img_index_str+"/y,"+str(y)) 
	file.write("\n")
	file.write(folder_str+"/"+img_index_str+"/z,"+str(z)) 
	file.write("\n")

header = ['guid','u_a','v_a']
inf = pd.read_csv('detection.csv',header = 0)
inf.columns = header
guid = inf['guid']
u_a = inf['u_a']
v_a = inf['v_a']

file = open("guid.txt", "w") 
file.write("guid/image/axis,value\n") 

files = glob('*/*_image.jpg')
for guidindex in files:
	folder = guidindex[:-15]
	img = guidindex[-14:-10]
	write(folder,img,guid,u_a,v_a)


