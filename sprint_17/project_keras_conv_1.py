from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
optimizer = Adam(learning_rate=.00002)
from matplotlib import pyplot as plt

def load_train(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    train_datagen_flow = datagen.flow_from_dataframe(dataframe=pd.read_csv(path+'labels.csv'),
    directory='/datasets/faces/final_files/',
    x_col='file_name',
    y_col='real_age',
    target_size=(224, 224),
    batch_size=16,
    class_mode='raw',
    seed=2807, 
    subset='training') 
    
    return train_datagen_flow 

def load_test(path):
    datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    test_datagen_flow = datagen.flow_from_dataframe(dataframe=pd.read_csv(path+'labels.csv'),
    directory='/datasets/faces/final_files/',
    x_col='file_name',
    y_col='real_age',
    target_size=(224, 224),
    batch_size=16,
    class_mode='raw',
    seed=2807, 
    subset='validation') 
    
    return test_datagen_flow

def create_model(input_shape):    
    model = Sequential()
    model.add(Conv2D(filters=6,
                 kernel_size=(5, 5),
                 padding='same',
                 activation='relu',
                 input_shape=(input_shape)))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                 activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=26, kernel_size=(5, 5), padding='valid',
                 activation="relu"))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1, activation='relu'))
    model.compile(loss='mean_squared_error', 
              optimizer='adam', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
    return model 


def train_model(model, train_data, test_data, batch_size=None, epochs=10,
                steps_per_epoch=None, validation_steps=None):
    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=len(train_data),
              validation_steps=len(test_data),
              verbose=2)
    return model

