import pandas as pd
import numpy as np
import keras

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
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
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        brightness_range=(0.3,1.5),
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = datagen.flow_from_directory(
        'data2/train',  # this is the target directory
        batch_size=16,
        class_mode='categorical')
    validation_generator = train_datagen.flow_from_directory(
        'data2/val',
        batch_size=16,
        class_mode='categorical')
    '''

    #model =  build_model(train_generator,validation_generator)
    model = load_model("best_model_multi_next.hdf5")
    counter0=0
    counter1=0
    counter2=0
    print('Opening the csv file')
    f = open('result_classification.csv',mode = 'w')
    rwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    rwriter.writerow(['guid/image', 'label'])
    print('Writing csv files')
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






    

def build_model(train_generator,validation_generator):
        file_path = "best_model_multi_next.hdf5"
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
        save_best_only = True, mode = "min")
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)
        '''
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
        '''
        model = keras.applications.xception.Xception(include_top=True, weights=None, input_tensor=None, input_shape=(256, 256, 3), pooling=None, classes=3)
        model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
        #history = model.fit(X_train, y, batch_size = 8, epochs = 40, validation_split=0.1,
        #verbose = 1, callbacks = [check_point, early_stop])
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=1000, # // batch_size = 1000,
            epochs=50,
            validation_data = validation_generator,
            validation_steps = 1000,
            verbose = 1, 
            callbacks = [check_point, early_stop])
            #class_weight=class_weight)
        model = load_model(file_path)

if __name__ == "__main__":
    main()

