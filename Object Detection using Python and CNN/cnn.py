# -*- coding: utf-8 -*-
"""Object Detection using Python and CNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lUthBuNGzUcOuCf3LJFxalCoIr1QfM3M
"""

!pip install keras-tuner  #conataining parameter any deep learning application

import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist= keras.datasets.fashion_mnist  #dataset contains different types of fashion items

(train_images,train_labels),(test_images,test_labels ) = fashion_mnist.load_data()

train_images= train_images/255.0
test_images=test_images/255.0

train_images[0].shape

train_images=train_images.reshape(len(train_images),28,28,1)
test_images=test_images.reshape(len(test_images),28,28,1)

def build_model(hp):

   model =keras.Sequential([  #any sequentional type of deep learning model where one thing follow the another in sequentional pattern # we use keras.Sequential 
                            keras.layers.Conv2D
                            (filters=hp.Int('conv_1_filter',min_value=32,max_value=128,step=16), # this line defines our kernal, the kernal that are useful conversion
                            kernel_size=hp.Choice('conv_2_kernel',values=[3,5]),  # kernal size either 3+3 or 5+5
                            activation='relu',
                            input_shape=(28,28,1)
   ),
   keras.layers.Conv2D
                       (filters = hp.Int('conv_2_filter',min_value=32,max_value=64,step=16),
                       kernel_size=hp.Choice('conv_2_kernel',values=[3,5]),
                       activation='relu'
   ),
                       keras.layers.Flatten(),
                       keras.layers.Dense(
                       units=hp.Int('dense_1_units',min_value=32,max_value=128,step=16), 
                       activation= 'relu'    
   ),
   keras.layers.Dense(10,activation='softmax')#ouputlayer
   
   ])
   model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-3])), #there are many optimizers but we use Adam(good choice) for its Good Accuracy 
                loss ='sparse_categorical_crossentropy',
                metrics =['accuracy'])
   return model

from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,directory='output',project_name="Mnist Fashion")

tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)

model = tuner_search.get_best_models(num_models=1)[0]

model.summary()
