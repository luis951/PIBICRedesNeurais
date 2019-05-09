# -*- coding: utf-8 -*-
"""
Arquivo de teste com Rede Neural Simples
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1) 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.elu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

model.save("modelo_teste.model")
modeloTeste = tf.keras.models.load_model("modelo_teste.model")

predictions = modeloTeste.predict(x_test)