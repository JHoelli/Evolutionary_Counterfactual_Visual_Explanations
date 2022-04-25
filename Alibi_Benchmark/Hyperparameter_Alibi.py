#!/usr/bin/env python
# coding: utf-8
import pandas as pd
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
import argparse
import sys
print(tf.__version__)
###########################This File Gridsearches differemt setting################################
##########################Identified Relevant Parameters:##########################################
#########################MutProb, CrossoverRate, Epochs, Population Size###########################
########################Indicator Picture Distance and Predition Probability######################################################

parser = argparse.ArgumentParser(description='TargetClass.')
parser.add_argument('dataset', metavar='data', type=str, nargs='+',help='MNIST or Fashion')
parser.add_argument('approach', type=str, nargs='+',help='Wachter or VanLooveren')

args = parser.parse_args()

#################################################################################################################################
########################Datatset Initalization###################################################################################
#################################################################################################################################

if args.dataset[0] == 'MNIST':
    model='../Model/cnn_MNIST.h5'
    cnn=tf.keras.models.load_model('../Model/cnn_MNIST.h5')
    data="mnist"
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
elif args.dataset[0] == 'Fashion':
    model = '../Model/cnn_Fashion_mnist.h5'
    data = "fashion"
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
else:
    print('No Valid dataset was found for tuning.')
    sys.exit(1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)

#cnn = load_model(model)
##################################################################################
###############Initialize Random Search ##########################################
#################################################################################
random.seed(1000)
number_runs= 30
parameters=[]
picture = random.randint(0, 10000)
##################################################################################
###############Parameters Identical for both Methods ##########################################
#################################################################################
shape = (1,) + x_train.shape[1:]
feature_range = (x_train.min(), x_train.max())
max_iter = 1000

if args.approach[0] == 'Wachter':
    #Fixed Parameters
    shape = (1,) + x_train.shape[1:]
    target_proba = 1.0
    target_class = 'other'  # any class other than 7 will do
    #Mainly optimization
    header=['lam_init','max_lam_steps','learning_rate_init','tol']
    lam_init = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10] #according to doc / use Paper default from Wachter?
    max_lam_steps = 10
    learning_rate_init = [0.1,0.01,1e-3, 1e-4 , 1e-5]
    tol = 0.01  # want counterfactuals with p(class)>0.99
    for a in range(0, 30):
        parameters.append([lam_init[random.randint(0, len(lam_init)-1)], 10*random.randint(1,10),learning_rate_init[random.randint(0, len(learning_rate_init)-1)],random.randint(0,49)*0.01])
elif args.dataset[0] == 'VanLooveren':
    # TODO: set Parameter Ranges
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
    autoencoder.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
    autoencoder.save(f'../Model_{args.dataset[0]}_ae.h5', save_format='h5')
    encoder.save(f'../Model_{args.dataset[0]}_enc.h5', save_format='h5')
    print('To_Implement')
    #TODO Parameter Ranges
    gamma = 100.
    theta = 100.
    c_init = 1. #weight on Prediction Loss term
    c_steps = 2

##############################################################################################################
##############################################################################################################
##############################################################################################################

#if not os.path.exists(str(num)):
#    os.makedirs(str(num))
results=[]
print(parameters)

for a in parameters:
    X = x_test[picture].reshape((1,) + x_test[1].shape)
    #print(X)
    print(a)
    print('Start')
    if args.approach[0] == 'Wachter':
        cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=a[3],
                    target_class=target_class, max_iter=max_iter, lam_init=a[0],
                    max_lam_steps=a[1], learning_rate_init=a[2],
                    feature_range=feature_range)
    elif args.approach[0] == 'VanLooveren':
        cf = CounterFactualProto(cnn, shape, gamma=gamma, theta=theta,
                             ae_model=autoencoder, enc_model=encoder, max_iterations=max_iterations,
                             feature_range=feature_range, c_init=c_init, c_steps=c_steps)

    try:
        print('Explain')
        explanation = cf.explain(X)
        #print(explanation)
        print(f'{a} finished')
        #print(1)
        pred_class = explanation.cf['class']
        #print(pred_class)
        #print(2)
        dis=np.mean(np.abs(np.subtract(X.reshape(-1), explanation['cf']['X'].reshape(-1))))
        #print(dis)
        #print(3)
        output_dis=1-explanation.cf['proba'][0][pred_class]
        #print(4)
        #do this with ranking
        results.append([dis,output_dis,(dis+output_dis)/2])
        #print(results)
    except:
        print('Not calculatable')
        results.append(['Result','was not',' calculateable'])
print(1)
list = np.concatenate((parameters, results), axis=1)
#print(2)
#index =np.where( list== np.min(list[:,-1]))
#print('Optimal Value:', list[index])
#print(list)
pd.DataFrame(list).to_csv(f'./{args.dataset[0]}_{args.approach[0]}_Hyperparameter.csv')
