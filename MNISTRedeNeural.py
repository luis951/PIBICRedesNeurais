# -*- coding: utf-8 -*-
"""
Arquivo de teste com Rede Neural Convolucional
"""

import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the input
x_train2= x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train2 = x_train2.astype('float32')
x_train2/=255
x_test2 = x_test.reshape(x_test.shape[0],28,28,1)
x_test2 = x_test2.astype('float32')
x_test2/=255

train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()

training_set= train_gen.flow(x_train2, y_train, batch_size=64)
test_set= train_gen.flow(x_test2, y_test, batch_size=64)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (2,2), activation='relu', input_shape=(28,28,1), 
                                 strides=1))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(12, (2,2), activation='relu', input_shape=(28,28,1), 
                                 strides=1))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit_generator(training_set, validation_data= test_set
                    , epochs=3, steps_per_epoch=60000/64, verbose=2)

model.save("modelo_teste.model")
modeloTeste = tf.keras.models.load_model("modelo_teste.model")

predictions = modeloTeste.predict(x_test2)