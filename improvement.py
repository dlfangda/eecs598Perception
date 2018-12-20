import pandas as pd
import numpy as np

from keras.models import Model, load_model


from glob import glob
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import OneHotEncoder
import csv

def main():
    detectionResult = pd.read_csv('fusion_detection.csv', engine='python')
    f = open('fusion_improvement.csv',mode = 'w')
    classResult = pd.read_csv('result_classification.csv', engine='python')
    rwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rwriter.writerow(['guid/image', 'label'])
    counter0=0
    counter1=0
    counter2=0
    for index, row in detectionResult.iterrows(): 
        result = row['label']
        if row['label'] == 0:
            index = row['guid/image'][37:]
            num = int(index)
            counts = [0,0,0]
            rowIndex = row['guid/image'][:-4]+conToString(num-1)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num-2)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num-3)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num-4)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1

            rowIndex = row['guid/image'][:-4]+conToString(num+1)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num+3)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num+4)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            rowIndex = row['guid/image'][:-4]+conToString(num+2)
            prev = detectionResult[detectionResult['guid/image'] == rowIndex]['label']
            if not prev.empty:
                counts[int(prev)] += 1
            if counts[1] == 0 and counts[2] == 0:
                #result = 0
                result = 0
            elif counts[1]>=3 and counts[1]>counts[2]:
                result = 1
                #result = int(classResult[classResult['guid/image'] == row['guid/image']]['label'])
            elif counts[2]>=3 and counts[2]>counts[1]:
                result = 2
                #result = int(classResult[classResult['guid/image'] == row['guid/image']]['label'])
            else:
                #result = 0
                result = int(classResult[classResult['guid/image'] == row['guid/image']]['label'])
        if result == 0: counter0 += 1
        if result == 1: counter1 += 1
        if result == 2: counter2 += 1
        rwriter.writerow([row['guid/image'], str(result)])


    print('Number of 0:',counter0)
    print('Number of 1:',counter1)
    print('Number of 2:',counter2)

def conToString(num):
    result = str(num)
    remSize = 4-len(result)
    for i in range(remSize):
        result = '0'+result
    return result

if __name__ == "__main__":
    main()