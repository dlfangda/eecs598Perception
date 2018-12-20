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
    classes = pd.read_csv('classes.csv', engine='python')
    labels = pd.read_csv('labels.csv', engine='python')
    centroids = pd.read_csv('centroids.csv', engine='python')
    testFiles = glob('test/*/*_image.jpg')
    trainFiles = glob('trainval/*/*_image.jpg')
    '''
    trainFiles = glob('deploy/trainval/*/*_image.jpg')
    shuffle(trainFiles)
    testFiles = glob('deploy/test/*/*_image.jpg')
    X_train = []
    label = []
    for file in trainFiles:
        if random.uniform(0, 1) > 0.8:
            img = load_img(file)
            x = img_to_array(img)
            X_train.append(x)
            curLabel = labels[labels['guid/image'] == file[16:-10]]['label']
            label.append(int(curLabel))
    X_test = []
    for file in testFiles:
        img = load_img(file)
        x = img_to_array(img)
        X_test.append(x)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    '''
    '''
    train_datagen = ImageDataGenerator(rescale=1./255)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(-0.7,2.0),
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = datagen.flow_from_directory(
        'data2/train',  # this is the target directory
        batch_size=16,
        class_mode='categorical')
    validation_generator = datagen.flow_from_directory(
        'data2/val',
        batch_size=16,
        class_mode='categorical')
     '''

    #model =  build_model(train_generator,validation_generator)
    model = load_model("best_model_multi.hdf5")
    counter0=0
    counter1=0
    counter2=0
    print('Opening the csv file')
    f = open('result_Xception_multi_test.csv',mode = 'w')
    rwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rwriter.writerow(['guid/image', 'label'])
    print('Reading csv files')
    '''

    testEvrs = glob('test/*/')
    for evr in testEvrs:
        files = glob(evr+'*_image.jpg')
        predictions = []

        for file in files:
            img = load_img(file,target_size=(256,256))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            predictions.append(model.predict(x))
        rwriter.writerow(files[0][5:-10], np.argmax(predictions[0]))
        rwriter.writerow(files[1][5:-10], np.argmax(predictions[1]))
        rwriter.writerow(files[len(files)-2][5:-10], np.argmax(predictions[len(files)-2]))
        rwriter.writerow(files[len(files)-1][5:-10], np.argmax(predictions[len(files)-1]))



        for i in range(len(file)-4):
            svector = predictions[i]+predictions[i+1]+predictions[i+2]+predictions[i+3]+predictions[i+4]
            result = np.argmax(svector)
            rwriter.writerow([file[5:-10], str(result)])
            
            if result == 0: counter0 += 1
            if result == 1: counter1 += 1
            if result == 2: counter2 += 1

    '''
    for file in testFiles:
        img = load_img(file,target_size=(256,256))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0
        y = model.predict(x)
        #decoded_preds = decode_predictions(preds)
        result = np.argmax(y)
        '''
        if result>=1 and result<=8:
            result = 1
        elif result>=9 and result<=14:
            result = 2
        else:
            result = 0
        '''

        if result == 0: counter0 += 1
        if result == 1: counter1 += 1
        if result == 2: counter2 += 1
        rwriter.writerow([file[5:-10], str(result)])
    print('Number of 0:',counter0)
    print('Number of 1:',counter1)
    print('Number of 2:',counter2)






    

if __name__ == "__main__":
    main()

