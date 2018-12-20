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
    detectionResult = pd.read_csv('result_detection_3.csv', engine='python')
    detectionResult2 = pd.read_csv('result_detection_4.csv', engine='python')
    f = open('fusion_detection.csv',mode = 'w')
    rwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rwriter.writerow(['guid/image', 'label'])
    counter0=0
    counter1=0
    counter2=0
    for index, row in detectionResult.iterrows(): 
        result = row['label']
        result2 = int(detectionResult2[detectionResult2['guid/image'] == row['guid/image']]['label'])
        if result == 0 and result2 == 0:
            result = 0
        elif result != 0 and result2 == 0:
            result = result2
        elif result == 0 and result2 != 0:
            result = result2
        else:
            result = result2
        
        if result == 0: counter0 += 1
        if result == 1: counter1 += 1
        if result == 2: counter2 += 1
        rwriter.writerow([row['guid/image'], str(result)])


    print('Number of 0:',counter0)
    print('Number of 1:',counter1)
    print('Number of 2:',counter2)

if __name__ == "__main__":
    main()