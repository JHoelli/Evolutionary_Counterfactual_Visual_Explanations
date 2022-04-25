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
#maybe interesting datasets : https://cvml.ist.ac.at/AwA/ , https://www.robots.ox.ac.uk/~vgg/data/flowers/random

def convolutional_Autoencoder(patches, train = True):
    #TODO save loss functions
    if train:
        p_train,p_test= train_test_split(patches)
        input = layers.Input(shape=(patches.shape[1], patches.shape[2], patches.shape[3]))

        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decoder
        x = layers.Conv2DTranspose(32, (3, 3),strides=(2,2), activation="relu", padding="same")(x)
        x=tf.keras.layers.UpSampling2D(size=(2, 2) )(x)

        x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
        #TODO Shape does not match
        x = layers.Conv2D(1, (3, 3), strides=(2,2), activation="relu", padding="same")(x) #used to be sigmoid

        # Autoencoder
        autoencoder = Model(input, x)
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.summary()
        autoencoder.fit( p_train,p_train,
            epochs=100,
            batch_size=128,
            shuffle=True,
            validation_data=(p_test, p_test)
        )

    return autoencoder, p_test

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
    #TODO Values ?
    '''
    Be aware Only Works with three channels
    @:param single_image : flattent starting picture
    @:param shape: shape of starting picture
    @:return augmented picture
    '''
    #if (shape[2]!=3):
        #TO RGB
    #    single_image.reshape(28,28)
    #    single_image=[single_image,single_image,single_image]
    # RandomContrast, RandomZoom, Random Crop --> Checkout here : https://www.tensorflow.org/tutorials/images/data_augmentation and here:
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing
    # keep some images original ? with less changes ?
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
    #TODO FLexibelise
    #shape=(28,28,1)
    #patch_size= 2
    #TODO How to do tis more efficiently
    ims = single_image
    im_per_side = np.sqrt(ims.shape[0]).astype(int)
    #print(im_per_side)
    #print(single_image.shape)

    # Reshape so we have real axes with desired orientation of images.
    #TODO make this to 1
    ims = np.reshape(ims, [im_per_side, im_per_side, single_image.shape[1], single_image.shape[2],shape[2] ])

    length_slices = []
    for l in range(im_per_side):
        height_slice = []
        for h in range(im_per_side):
            height_slice.append(ims[l, h])
        length_slices.append(np.concatenate(height_slice, 1))

    # Finally concatenate along the `length` axis.
    final_ims = np.concatenate(length_slices, 0)
    #print('FInal Image', final_ims.shape)

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
        #if shape[2]==3:
        #    if random.uniform(0, 1) > prob:
        #        single_image = tf.image.random_hue(single_image, 0.08)
        #    if random.uniform(0, 1) > prob:
        #        single_image = tf.image.random_saturation(single_image, 0.6, 1.6)
        single_image = np.clip(single_image, 0, 255)
        if (shape[2] != 3):
            single_image=(single_image[0,:,:,0]+single_image[0,:,:,1]+single_image[0,:,:,2])/3
            single_image=single_image.reshape(28,28,1)

            #TODO add a reshape ?
            #print('Test',single_image.shape)
    else:
        if (shape[2] != 3):
            single_image = np.array(single_image, dtype=np.int32)
        #single_image=single_image.reshape(28,28,1)
    if view_as_blog:
        M=shape[0]//blogs
        N=shape[1]//blogs #used to be 50
        #print(M)
        #single_image= [single_image[x:x + M, y:y + N].copy() for x in range(0, shape[0], M) for y in range(0, shape[1], N)]
        single_image=single_image.reshape(shape[0],shape[1],shape[2])
        #single_image=single_image.reshape(shape[0]//M, M, shape[1]//N, N, 3)
        #single_image=single_image.swapaxes(1,2)
        #print(single_image.shape)
        single_image = [single_image[x:x + M, y:y + N].copy() for x in range(0, shape[0], M) for y in
                       range(0, shape[1], N)]

        #for a in single_image:
        #    print(a.shape)
        #single_image=np.array(single_image)
        #print(type(single_image))
        #print(type(single_image[0]))
        #print(single_image[0].shape)
        #print(shape)
        #single_image=skimage.util.view_as_blocks(single_image.reshape(shape[0],shape[1],shape[2]),(3,3,shape[2]))
        #print(single_image.shape)
        #single_image=single_image.transpose(0, 2, 1, 3)
        #single_image=single_image.reshape(-1,3,3,shape[2])
        single_image=np.array(single_image)
        #print('!!!!!!!!',single_image[0].shape)
    #TODO CHange back to 9
    else:
        if (shape[2] != 3):
            single_image = sklearn.feature_extraction.image.extract_patches_2d(single_image, (12, 12), max_patches=max_patches)
        else:
            single_image = sklearn.feature_extraction.image.extract_patches_2d(single_image.reshape(shape[0],shape[1],shape[2]), (12, 12),max_patches=max_patches)
    if shuffle:
        #print('Try Shuffle')
        #TODO ssemed to be not working !!!
        #random.shuffle(np.array(single_image,dtype=np.int32))

        np.take(single_image, np.random.permutation(single_image.shape[0]), axis=0, out=single_image);
    return single_image


def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """
    #TODO devide by 255 ?
    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def hsv2rgb(hsv):
    """ convert HSV to RGB color space

    :param hsv: np.ndarray
    :return: np.ndarray
    """
    #TODO devide by 255 
    hi = np.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb

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

def load_chest_xray():
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        '/media/jacqueline/Data/CF_DATA/chest_xray/train/',
        target_size=(300, 300),
        batch_size=128,
        class_mode='categorical'  # 'binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        '/media/jacqueline/Data/CF_DATA/chest_xray/test/',
        target_size=(300, 300),
        batch_size=128,
        class_mode='categorical'  # 'binary'
    )


    X_test = []
    y_test = []
    for x, y in next(zip(validation_generator)):#validation_generator:
        for i in range(16):
            X_test.append(x[i])
            y_test.append(y[i])
    print(type(X_test))
    print(np.array(X_test).shape)
    print(np.array(y_test).shape)
    return train_generator,None, X_test, y_test
    #pass

def load_other_data(shape=(300,300)):
    #TODO Change Pathes
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory('/media/jacqueline/Data/CF_DATA/rscbjbr9sj-2/OCT2017/train/',
        target_size=(shape),
        batch_size=126,
        class_mode='categorical'  # 'binary'
    )

    test_generator = test_datagen.flow_from_directory('/media/jacqueline/Data/CF_DATA/rscbjbr9sj-2/OCT2017/test/',
        target_size=(shape),
        batch_size=126,
        class_mode='categorical'  # 'binary'
    )


    X_test = []
    y_test = []
    test_datagen = ImageDataGenerator()
    gen= test_datagen.flow_from_directory('/media/jacqueline/Data/CF_DATA/rscbjbr9sj-2/OCT2017/test/',
                                       target_size=(shape),
                                       batch_size=126,
                                       class_mode='categorical',
                                          shuffle = True,# 'binary'
                                          seed=15#13->1#10-->1#2 #prev 0
                                       )
    for x, y in next(zip(gen)):#validation_generator:
        for i in range(16):
            X_test.append(x[i])
            y_test.append(y[i])
    print(type(X_test))
    print(np.array(X_test).shape)
    print(np.array(y_test).shape)
    return train_generator, test_generator, X_test, y_test


def load_caltech_bird(size=(150,150,3)):
    #TODO Testing
    # https://www.tensorflow.org/guide/keras/transfer_learning Chapter data augmentation

    train_ds, validation_ds, test_ds = tfds.load(
        "caltech_birds2011",
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
    # train_ds = train_ds.map(lambda x, y: (tf.divide(x,255), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    # validation_ds = validation_ds.map(lambda x, y: (tf.divide(x,255), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    # test_ds = test_ds.map(lambda x, y: (tf.divide(x,255), y))
    train_ds = train_ds.map(lambda x, y: mask_to_categorical(x, y))
    validation_ds = validation_ds.map(lambda x, y: mask_to_categorical(x, y))
    test_ds = test_ds.map(lambda x, y: mask_to_categorical(x, y))
    # print(train_ds)
    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    # test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    # test_x, test_y=test_ds
    # print(test_y)
    # TODO Map to train / Test split --> not necessary as .predict oly needs data?
    # test_ds=list(tfds.as_numpy(test_ds))
    X_test = []
    y_test = []
    for x, y in test_ds:
        X_test.append(x.numpy())
        y_test.append(y.numpy())
    # print(type(X_test))

    # print(X_test)
    return train_ds, validation_ds, X_test, y_test



def load_horse_vs_zebra():
    # https://www.tensorflow.org/guide/keras/transfer_learning Chapter data augmentation

    dataset = tfds.load(
        "cycle_gan/horse2zebra",
        as_supervised=True,  # Include labels
    )
    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    size = (150, 150)
    #size=(224,224) # Because we use VGG16
    train_horses = train_horses.map(lambda x, y: (tf.image.resize(x, size), y))
    train_zebras = train_zebras.map(lambda x, y: (tf.image.resize(x, size), y))
    train_horses = train_horses.map(lambda x, y: mask_to_categorical(x,y))
    train_zebras= train_zebras.map(lambda x, y: mask_to_categorical(x, y))
    test_horses = test_horses.map(lambda x, y: (tf.image.resize(x, size), y))
    test_zebras = test_zebras.map(lambda x, y: (tf.image.resize(x, size), y))
    test_horses = test_horses.map(lambda x, y: mask_to_categorical(x, y))
    test_zebras = test_zebras.map(lambda x, y: mask_to_categorical(x, y))
    X_test= []
    y_test=[]
    X_train = []
    y_train = []
    for x , y in train_horses:
        X_train.append(x.numpy()/255)
        y_train.append(y.numpy())
    for x,y in train_zebras:
        X_train.append(x.numpy()/255)
        y_train.append(y.numpy())

    c = list(zip(X_train, y_train))

    random.shuffle(c)

    X_train, y_train = zip(*c)

    for x , y in test_horses:
        X_test.append(x.numpy()/255)
        y_test.append(y.numpy())
    for x,y in test_zebras:
        X_test.append(x.numpy()/255)
        y_test.append(y.numpy())

    c = list(zip(X_test, y_test))

    random.shuffle(c)

    X_test, y_test = zip(*c)

    print(f'Xtrain number samples {len(X_train)}')
    print(f'Xtrain shape {X_train[0].shape}')
    print(f'Xtest number samples {len(X_test)}')
    print(f'Xtest shape {X_test[0].shape}')

    print(f'ytrain number samples {len(y_train)}')
    print(f'ytrain shape {y_train[0].shape}')
    print(f'ytest number samples {len(y_test)}')
    print(f'ytest shape {y_test[0].shape}')
    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)




def load_cal_tech_birds():
    #TODO still to write
    #download https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view/CUB_200_2011.tgz
    #http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    (trainX, trainY), (testX, testY) = tf.Data.Dataset.image_classification.caltech_birds2011()
    #tf.keras.datasets.caltech_birds2011.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

#TODO change
def plot_patches_for_one_image(patches,patch_size,n=30, figsize=15,save= None):
    #used to be 3
    #TODO simplyfy, patchsize used to be 12
    # display a n*n 2D manifold of digits
    digit_size = patch_size
    #Currently 7 for 28 / 28
    #digit_size=2
    print(digit_size)
    patches = np.array(patches)
    #try:
    #    patches = patches.reshape(-1,digit_size, digit_size,3)
    #except:
    #TODO make this formallly correct
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


if __name__ == "__main__":
    #Test for Data Augmentation with patches
    #x_train, y_train,x_test, y_test=load_mnist()
    #for i in range(0,5):
    #x= x_train[0]
    #recon=data_augmentation_patch_based(x,(150,150,1))
    #print(recon.shape)
    #recon=recon.reshape(-1,12,12,1)
    #plot_patches_for_one_image(recon,save='original.png')
    #model, p_test = convolutional_Autoencoder(recon)
    #new_recon=model.predict(p_test)
    #recon=sklearn.feature_extraction.image.reconstruct_from_patches_2d(recon,image_size=(28,28))
    #plt.subplot(211)
    #plt.imshow(p_test[0].reshape(12,12))
    #plt.subplot(212)
    #plt.imshow(new_recon[0].reshape(12, 12))
    #plt.show()
    train_generator, test_generator, X_test, y_test=load_other_data()
    #import tensorflow as tf
    #model=tf.keras.models.load_model('./Model/TrainingModels/callback')
    #print(model.evaluate(test_generator))
    #print(np.array(X_test).shape)
    i=0
    for pic in range(0, len(X_test)):
        plt.figure()
        plt.imshow(np.array(X_test[pic]/255).reshape(300,300,3))
        plt.savefig(f'Im_{i}.png')
        i=i+1
    #load_chest_xray()
    #Checkout if - values come from Data Augmentation or Patches
    #for a in range(0,10):
    #    image =data_augmentation_patch_based(x,(28,28,1), True)#data_augmentation(x, (28,28,1))
    #    print(image)
    #    image = sklearn.feature_extraction.image.reconstruct_from_patches_2d(image, image_size=(28, 28))
    #    print(image)
    #    t=image[image < 1]
    #    #print(t)
    #    print(len(t))
    #    #print(t)
    #    if len(t)!= 0:
    #        if len(t[t > 0])>0:
    #            print('trigger checks for double values ')
    #        if len(t[t < 0]) > 0:
    #            print('negative value should not be happening')

    #Load Color Dataset
    #Make Patches
    #Show Patches
    #Train Model
    #Show loss
    #Show Reconstruction
    #Reconstruct Full Image
