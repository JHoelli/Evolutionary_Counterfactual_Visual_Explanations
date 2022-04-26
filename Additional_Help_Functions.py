import math

import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import numpy as np
from itertools import product
import random
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn.feature_extraction
from sklearn.model_selection import train_test_split

def reconstruct_from_patches_2d(patches, image_size):
    '''Wrapper to cast sklearn in shuffeld case to Int --> should not have an influence for the none shuffeld case'''
    #print(patches.shape)
    patches = np.array(patches)
    try:
        img=sklearn.feature_extraction.image.reconstruct_from_patches_2d(patches,image_size)
    except:
        img = sklearn.feature_extraction.image.reconstruct_from_patches_2d(patches, (image_size[0], image_size[1]))
    img= np.array(img)
    img=np.around(img,0)
    return img

def plot_images(pop, shape,num_pics,procid):
    print(f'Start Plot for {procid}')
    total = num_pics
    num_pics = math.sqrt(num_pics)
    num_pics=math.ceil(num_pics)
    plt.figure()
    for i in range(0,total):
        #print(i)
        pic= np.array(pop[i]).reshape(shape[0],shape[1], shape[2])
        #print(pic.shape)
        #print(pic)
        plt.subplot(num_pics, num_pics, i + 1)
        plt.imshow(pic)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'Population_{procid}.png')

def plot_patches(pop, shape,num_pics,procid,patch_size=6,and_rest=False):
    print(f'Start Plot for {procid}')
    plot_patches_for_one_image(pop[0],n=30,figsize=15, save =f'Population_Patches_{procid}.png',patch_size=patch_size )
    print(f'Patches Plot FInished for {procid}')
    total = num_pics
    num_pics = math.sqrt(num_pics)
    plt.figure()
    for i in range(0, total):
        # print(i)
        #print(shape)
        #print(np.array(pop[i]).shape)
        #pic=reconstruct_from_patches_2d(np.array(pop[i]), image_size=shape)
        #pic=sklearn.feature_extraction.image.reconstruct_from_patches_2d(np.array(pop[i]), shape)
        pic= np.array(pop[i])
        #pic=pic.transpose(0, 2, 1, 3).reshape(-1, shape[1] * shape[3])
        try:
            #pic = sklearn.feature_extraction.image.reconstruct_from_patches_2d(pic, shape)
            pic= reshape_to_image(pic)
            #pic = np.array(pic).reshape(-1,shape[0], shape[1], shape[2])
            print('Pic',pic.shape)
        except:
            pic = np.array(pic).reshape(shape[0], shape[1])
        # print(pic.shape)
        # print(pic)
        plt.subplot(num_pics, num_pics, i + 1)
        plt.imshow(pic[0])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'Population_{procid}.png')
        print(f'Plot for {procid} was created')
    return 'Nothing'



def load_cifar():
    # load dataset
    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
    testX = testX.reshape((testX.shape[0], 32, 32, 3))
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

def randomImage(data_set,y,procid):
    #trainX, trainY, testX, testY = load_dataset()
    ids= np.where(y==procid)
    #print('ID ',ids)
    data_set= data_set[ids[0][random.randint(0, len(ids))]].reshape(-1,32*32*3)
    #print(data_set[0])
    return data_set[0]

def singleImage(single_image):
    return single_image.reshape(-1)

def data_augmentation(single_image, shape):
    '''
    Be aware Only Works with three channels
    @:param single_image : flattent starting picture
    @:param shape: shape of starting picture
    @:return augmented picture
    '''
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomContrast(factor=(0.1, 1.3)),
        layers.experimental.preprocessing.RandomZoom(height_factor=(-0.7, -0.2)),
    ])
    single_image=np.array(single_image).reshape(1,shape[0], shape[1], shape[2])
    #single_image=single_image.astype(int)
    prob=0.5
    single_image=data_augmentation(single_image)
    if shape[2]==3:
        if random.uniform(0, 1) > prob:
            single_image = tf.image.random_hue(single_image, 0.08)
        if random.uniform(0, 1) > prob:
            single_image = tf.image.random_saturation(single_image, 0.6, 1.6)
    single_image = np.clip(single_image, 0, 255)
    #if (shape[2] != 3):
    #    single_image=(single_image[0,:,:,0]+single_image[0,:,:,1]+single_image[0,:,:,2])/3
    return np.array(single_image,dtype=np.int32).reshape(-1)

def reshape_to_image(single_image, shape=(150,150,3)):
    ims = single_image
    im_per_side = np.sqrt(ims.shape[0]).astype(int)
    ims = np.reshape(ims, [im_per_side, im_per_side, single_image.shape[1], single_image.shape[2],shape[2] ])

    length_slices = []
    for l in range(im_per_side):
        height_slice = []
        for h in range(im_per_side):
            height_slice.append(ims[l, h])
        length_slices.append(np.concatenate(height_slice, 1))

    # Finally concatenate along the `length` axis.
    final_ims = np.concatenate(length_slices, 0)

    return final_ims.reshape(1,shape[0],shape[1],shape[2]) #image

def data_augmentation_patch_based(single_image, shape, data_augmentation = False,shuffle=True, max_patches=None,view_as_blog=True,blogs=10): #TODO Used to be 4
    '''
    Be aware Only Works with three channels
    @:param single_image : flattent starting picture
    @:param shape: shape of starting picture
    @:param data_augmentation: use data augmentation
    @:return augmented picture patches
    '''
    #print(single_image.shape)
    if data_augmentation:
        if (shape[2]!=3):
            #TO RGB
            single_image.reshape(28,28)
            single_image=[single_image,single_image,single_image]
        data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.2),
            #layers.experimental.preprocessing.RandomContrast(factor=(0.1, 1.3)),
            layers.experimental.preprocessing.RandomZoom(height_factor=(-0.7, -0.2)),
        ])
        single_image=np.array(single_image).reshape(1,shape[0], shape[1], 3)
        single_image=single_image.astype(int)
        prob=0.5
        single_image=data_augmentation(single_image)
        single_image = np.clip(single_image, 0, 255)
        if (shape[2] != 3):
            single_image=(single_image[0,:,:,0]+single_image[0,:,:,1]+single_image[0,:,:,2])/3
            single_image=single_image.reshape(28,28,1)

    else:
        if (shape[2] != 3):
            single_image = np.array(single_image, dtype=np.int32)
        #single_image=single_image.reshape(28,28,1)
    if view_as_blog:
        M=shape[0]//blogs
        N=shape[1]//blogs #used to be 50

        single_image=single_image.reshape(shape[0],shape[1],shape[2])
        single_image = [single_image[x:x + M, y:y + N].copy() for x in range(0, shape[0], M) for y in
                       range(0, shape[1], N)]

        single_image=np.array(single_image)

    else:
        if (shape[2] != 3):
            single_image = sklearn.feature_extraction.image.extract_patches_2d(single_image, (12, 12), max_patches=max_patches)
        else:
            single_image = sklearn.feature_extraction.image.extract_patches_2d(single_image.reshape(shape[0],shape[1],shape[2]), (12, 12),max_patches=max_patches)
    if shuffle:

        np.take(single_image, np.random.permutation(single_image.shape[0]), axis=0, out=single_image);
    return single_image


def load_mnist():
    # load dataset
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

def load_fashion_mnist():
    # load dataset
    (trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

def mask_to_categorical(image, mask):
    mask = tf.one_hot(tf.cast(mask, tf.int32), 2)
    mask = tf.cast(mask, tf.float32)
    return image, mask

def seperate_x_y(x,y):
    return (x,y)

def load_cat_vs_dog(size=(150,150)):
    # https://www.tensorflow.org/guide/keras/transfer_learning Chapter data augmentation

    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        # Reserve 10% for validation and 10% for test
        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
        as_supervised=True,  # Include labels
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print(
        "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
    )
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))


    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    #train_ds = train_ds.map(lambda x, y: (tf.divide(x,255), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    #validation_ds = validation_ds.map(lambda x, y: (tf.divide(x,255), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    #test_ds = test_ds.map(lambda x, y: (tf.divide(x,255), y))
    train_ds = train_ds.map(lambda x, y: mask_to_categorical(x,y))
    validation_ds = validation_ds.map(lambda x, y: mask_to_categorical(x, y))
    test_ds = test_ds.map(lambda x, y: mask_to_categorical(x, y))
    #print(train_ds)
    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    #test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    #test_x, test_y=test_ds
    #print(test_y)
    # TODO Map to train / Test split --> not necessary as .predict oly needs data?
    #test_ds=list(tfds.as_numpy(test_ds))
    X_test= []
    y_test=[]
    for x , y in test_ds:
        X_test.append(x.numpy())
        y_test.append(y.numpy())
       # print(type(X_test))

    #print(X_test)
    return train_ds, validation_ds,X_test,y_test

def plot_patches_for_one_image(patches,patch_size,n=30, figsize=15,save= None):
    digit_size = patch_size
    print(digit_size)
    patches = np.array(patches)
    try:
        patches = patches.reshape(-1, digit_size, digit_size,1)
    except:
        patches = patches.reshape(-1, digit_size, digit_size, 3)
    print(patches.ndim)

    if patches.ndim != 4:
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                digit = patches[i+j].reshape(digit_size, digit_size)
                figure[
                    i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size,
                    ] = digit
        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.imshow(figure)
    else:
        dpi = 100
        plot_width, plot_height = 1200, 800
        width_inches, height_inches = plot_width / dpi, plot_height / dpi
        # Show patches in a 10x10 grid
        if patches.shape[0]>15*15:
            gridx, gridy = 15, 15
        else:
            gridx, gridy = 4, 4
        fig, ax = plt.subplots(figsize, figsize, figsize=(width_inches, height_inches), dpi=dpi, facecolor='w',
                               edgecolor='k', frameon=False)

        for i in range(gridx):
            for j in range(gridy):
                #im = np.array(patches[i + j].reshape(digit_size, digit_size, 3))
                im = np.array(patches[i + j].reshape(digit_size, digit_size, 1))
                ax[i, j].axis('off')
                ax[i, j].imshow(im)

    if save==None:
        plt.show()
    else:
        plt.savefig(save)

    return 'Nothing'
