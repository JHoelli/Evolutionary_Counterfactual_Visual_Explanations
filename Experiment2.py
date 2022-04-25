import sys
import random
import itertools
import numpy as np
import pandas as pd
import os
import random
import Additional_Help_Functions as helper
import argparse

parser = argparse.ArgumentParser(description='TargetClass.')
parser.add_argument('data', metavar='d', type=str, help='MNIST or Fashion')
args = parser.parse_args()
dataset = args.data

if dataset == 'MNIST':
    model = "./Model/cnn_MNIST.h5"
    data = "mnist"
    _,_,test,_=helper.load_mnist()
elif dataset == 'Fashion':
    model = "./Model/cnn_Fashion_mnist.h5"
    data = "fashion"
else:
    print('No Valid dataset was found.')
    sys.exit()
runs=30


mutation=['mutUniformInt']
parameters = []
pic=[]
image= np.unique(pd.read_csv('./Results/Distance/MNIST_ssim.csv')['image_number'])

for i in mutation:
    for p in image:
        parameters.append([i,p])

parameters = pd.DataFrame(parameters, columns=['mutation', 'image_number'])
parameters.to_csv(f'./Results/Mutation/{dataset}.csv')

for a in range(0, len(parameters)):
    image=str(parameters['image_number'][a])
    dis=parameters['mutation'][a]
    if not os.path.isdir(f'./Results/Mutation/{dataset}/{dis}/{image}'):
        script_descriptor = open("Evo_Patch_Level.py")
        a_script = script_descriptor.read()
        sys.argv = ["main.py", "ShufflePatches","14",str(parameters['image_number'][a]), data, model, "1", "Alt", "100",
                "1000", "0", "uniform", "auto2",str(parameters['mutation'][a]) ,"auto2",
                "NSGA2", "./Results/Mutation/" + str(dataset) + "/" +parameters['mutation'][a]+ "/",'ssim']
        exec(a_script)
        script_descriptor.close()
