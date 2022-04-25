import tensorflow as tf
 # suppress deprecation messages
#tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras import backend as K
import tensorflow
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from alibi.explainers import CounterFactual
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
    print('Load Model')
    cnn = tf.keras.models.load_model('../Model/cnn_Fashion_mnist.h5')
    print('Model Loaded')



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

path= f'../Results/Alibi/Wachter/{args.dataset[0]}/Direction_Other/'
if not os.path.exists(path+str(num)):
    os.makedirs(path+str(num))

X = x_test[num].reshape((1,) + x_test[1].shape)
shape = (1,) + x_train.shape[1:]
target_proba = 1.0
tol = 0.01 # want counterfactuals with p(class)>0.99
target_class = 'other' # any class other than 7 will do
max_iter = 1000
lam_init = 1e-1
max_lam_steps = 10
learning_rate_init = 0.1
feature_range = (x_train.min(),x_train.max())

output= cnn.predict(X)
output= np.argmax(output)
#print(output)
print('CF constructor')
cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)
print('CF constructed')
#print(cf)
start_time = time()
print(start_time)
explanation = cf.explain(X)
#print(explanation)
print('Explanation took {:.3f} sec'.format(time() - start_time))


# In[ ]:
import pickle

pred_class = explanation['cf']['class']
proba = explanation['cf']['proba'][0][pred_class]

print(f'Counterfactual prediction: {pred_class} with probability {proba}')

pickle.dump(explanation, open(path+str(num)+"/explain_noTarget.pkl", "wb"), -1)
