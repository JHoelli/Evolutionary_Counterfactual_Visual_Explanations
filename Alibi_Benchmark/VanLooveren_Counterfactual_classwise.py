#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
 # suppress deprecation messages
#tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras import backend as K
import tensorflow
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input,UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from alibi.explainers import CounterFactual,CounterFactualProto
import random
random.seed(1000)
import argparse
print(tf.__version__)
parser=argparse.ArgumentParser(description='Picture to Process')
parser.add_argument('picnumb', metavar='i', type=int, nargs='+')
parser.add_argument('dataset',type=str,nargs='+')
args=parser.parse_args()

if args.dataset[0] == 'MNIST':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    cnn = tf.keras.models.load_model('../Model/cnn_MNIST.h5')

if args.dataset[0] == 'Fashion':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    cnn = tf.keras.models.load_model('../Model/cnn_Fashion_mnist.h5')



# In[3]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

num= args.picnumb[0]

path= f'../Results/Alibi/VanLooveren/{args.dataset[0]}/Classwise/'
if not os.path.exists(path+str(num)):
    os.makedirs(path+str(num))


shape = (1,) + x_train.shape[1:]
X = x_test[num].reshape((1,) + x_test[1].shape)
output= cnn.predict(X)
output= np.argmax(output)

def ae_model():
    # encoder
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    encoder = Model(x_in, encoded)

    # decoder
    dec_in = Input(shape=(14, 14, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_in)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    decoder = Model(dec_in, decoded)

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder

if not os.path.exists(f'../Model/{args.dataset[0]}_ae.h5'):
    ae, enc, dec = ae_model()
    ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
    ae.save(f'../Model/{args.dataset[0]}_ae.h5', save_format='h5')
    enc.save(f'../Model/{args.dataset[0]}_enc.h5', save_format='h5')


ae = load_model(f'../Model/{args.dataset[0]}_ae.h5')
enc = load_model(f'../Model/{args.dataset[0]}_enc.h5', compile=False)


shape = (1,) + x_train.shape[1:]
gamma = 100.
theta = 100.
c_init = 1.
c_steps = 2
max_iterations = 1000
feature_range = (x_train.min(),x_train.max())

cf = CounterFactualProto(cnn, shape, gamma=gamma, theta=theta,
                         ae_model=ae, enc_model=enc, max_iterations=max_iterations,
                         feature_range=feature_range, c_init=c_init, c_steps=c_steps)

start_time = time()
cf.fit(x_train)  # find class prototypes
print('Time to find prototypes each class: {:.3f} sec'.format(time() - start_time))
start_time = time()


if not os.path.exists(path+str(num)):
    os.makedirs(path+str(num))
for a in range(0,10):
    if(a!=output):


        target_class = a # any class other than 7 will do
        start_time = time()
        print(start_time)
        try:
            explanation = cf.explain(X,target_class=[a])
            print(explanation)
            print('Explanation took {:.3f} sec'.format(time() - start_time))


# In[ ]:
            import pickle

            if not os.path.exists(path+str(num) + '/' + str(a)):
                os.makedirs(path+ str(num) + '/' + str(a))
            pred_class = explanation['cf']['class']
            proba = explanation['cf']['proba'][0][pred_class]

            print(f'Counterfactual prediction: {pred_class} with probability {proba}')

            pickle.dump(explanation, open(path+str(num)+"/"+str(target_class)+"/explain_Target.pkl", "wb"), -1)
        except:
            print('fail')