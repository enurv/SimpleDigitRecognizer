# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:50:37 2019

@author: Eda
"""
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=2)

#plt.imshow(x_test[123])
#plt.show()
#prediction = model.predict(np.array([x_test[123],]))
#print(np.argmax(prediction))

pencereAdi = 'Canvas'
img = np.zeros((256,256))
cv2.namedWindow(pencereAdi)

cizimAktif = False

def firca(event, x, y, flags, param):
    global cizimAktif
    if event == cv2.EVENT_LBUTTONDOWN:
        cizimAktif = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cizimAktif == True:
            cv2.circle(img, (x, y), 10, 1, -1)
    elif event== cv2.EVENT_LBUTTONUP:
        cizimAktif = False
        cv2.circle(img, (x, y), 10, 1, -1)

cv2.setMouseCallback(pencereAdi, firca)

def main():
    while(True):
        cv2.imshow(pencereAdi, img)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    resized = cv2.resize(img, (28, 28))
    prediction = model.predict(np.array([resized,]))
    print(np.argmax(prediction))
        
if __name__ == "__main__":
    main()
    

#model.save('myFirst.model')
