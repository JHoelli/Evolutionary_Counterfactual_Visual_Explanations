import sys
import random
import itertools
import pandas as pd
import os
import random
import Additional_Help_Functions as helper

runs=15
import argparse

parser = argparse.ArgumentParser(description='TargetClass.')
parser.add_argument('data', metavar='d', type=str, help='MNIST or Fashion')
args = parser.parse_args()
dataset = args.data

print('DATASET',dataset)
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


dis_function=['ssim','issm','fsim','me','rmse']
parameters = []
pic=[]
for a in range(0, runs):
    pic.append(random.randint(0,len(test)))
for i in dis_function:
    for p in pic:
        parameters.append([i,p])

parameters = pd.DataFrame(parameters, columns=['distance', 'image_number'])
parameters.to_csv(f'./Results/Distance/{dataset}.csv')
print(parameters)
for a in range(0, len(parameters)):
    image=str(parameters['image_number'][a])
    dis=parameters['distance'][a]
    if not os.path.isdir(f'./Results/Distance/{dataset}/{dis}/{image}'):
        script_descriptor = open("Evo_Patch_Level.py")
        print("start run ")
        a_script = script_descriptor.read()
        # TODO Popluation 100 or 1000
        sys.argv = ["main.py", "ShufflePatches","14",str(parameters['image_number'][a]), data, model, "1", "Alt", "100",
                "1000", "0", "uniform", "auto2", "randomAugmentedPatch","auto2",
                "NSGA2", "./Results/Distance/" + str(dataset) + "/" +parameters['distance'][a]+ "/",parameters['distance'][a]]
        exec(a_script)
        script_descriptor.close()
