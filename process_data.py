import pandas as pd
import numpy as np

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from shutil import copy2
import os
import random
import array


def main():
    labels = pd.read_csv('labels.csv', engine='python')
    centroids = pd.read_csv('centroids.csv', engine='python')

    testFiles = glob('test/*/*_image.jpg')
    trainFiles = glob('trainval/*/*_image.jpg')
    idx = np.random.randint(0, len(trainFiles))


    X_train = []
    label = []
    os.makedirs('./data2',exist_ok=True)

    os.makedirs('./data2/train',exist_ok=True)
    os.makedirs('./data2/val',exist_ok=True)
    '''

    for index in range(23):
        os.makedirs('./data/train/'+chr(ord('a') + index),exist_ok=True)
        os.makedirs('./data/val/'+chr(ord('a') + index),exist_ok=True)
    '''

    directory = './data2'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/train'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/train/1'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/train/2'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/train/0'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/val'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/val/1'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/val/2'
    os.makedirs(directory,exist_ok=True)
    directory = './data2/val/0'
    os.makedirs(directory,exist_ok=True)



    index = 0
    for file in trainFiles:
        if random.uniform(0, 1) > 0.9:
            curLabel = labels[labels['guid/image'] == file[9:-10]]['label']
            directory = './data2/val/'+str(int(curLabel))+'/'+ str(index) + '.jpg'
            copy2(file,directory)
            '''
            if int(curLabel) == 0:
                directory = './data/val/'+str(chr(ord('a') + 0))+'/'+ str(index) + '.jpg'
                copy2(file,directory)
            else:

                try:
                    bbox = np.fromfile(file.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
                except FileNotFoundError:
                    print('Error Here')
                bbox = bbox.reshape([-1, 11])
                classid = int(bbox[0][9])
                directory = './data/val/'+str(chr(ord('a') + classid))+'/'+ str(index) + '.jpg'
                copy2(file,directory)
                '''
        else:
            curLabel = labels[labels['guid/image'] == file[9:-10]]['label']
            directory = './data2/train/'+str(int(curLabel))+'/'+ str(index) + '.jpg'
            copy2(file,directory)
            '''
            if int(curLabel) == 0:
                directory = './data/train/'+str(chr(ord('a') + 0))+'/'+ str(index) + '.jpg'
                copy2(file,directory)
            else:

                try:
                    bbox = np.fromfile(file.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
                except FileNotFoundError:
                    print('Error Here')
                bbox = bbox.reshape([-1, 11])
                classid = int(bbox[0][9])
                directory = './data/train/'+str(chr(ord('a') + classid))+'/'+ str(index) + '.jpg'
                copy2(file,directory)
                '''

        #else:

            #curLabel = labels[labels['guid/image'] == file[9:-10]]['label']
            #directory = './data/val/'+str(int(curLabel))+'/'+ str(index) + '.jpg'
            #copy2(file,directory)
        #else:
            #curLabel = labels[labels['guid/image'] == file[9:-10]]['label']
            #directory = './data/train/'+str(int(curLabel))+'/'+ str(index) + '.jpg'
            #copy2(file,directory)
        index += 1

if __name__ == "__main__":
    main()

