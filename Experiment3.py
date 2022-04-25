import argparse
import os
import sys
parser = argparse.ArgumentParser(description='TargetClass.')
parser.add_argument('data', metavar='d', type=str, help='MNIST or Fashion')
args = parser.parse_args()
data = args.data

if data == 'MNIST':
    img = [0,3,4,5,7,8,18,21,43, 232]
    model = "./Model/cnn_MNIST.h5"
    data = "mnist"

elif data =='Fashion':
    img = [0,1,2,4,6,9,11,13,18,19]
    model = "./Model/cnn_Fashion_mnist.h5"
    data = "fashion"

for i in img:
    if not os.path.isdir(f'./Results/Benchmark/Evo/{data}/{i}'):
        script_descriptor = open("Evo_Patch_Level.py")
        print("start run ")
        a_script = script_descriptor.read()
        sys.argv = ["main.py", "ShufflePatches", "14", str(i), data, model, "10", "Alt",
                    "500",
                    "1000", "0", "uniform", "auto2", 'randomAugmentedPatch', "auto2",
                    "NSGA2", "./Results/Benchmark/Evo/" + str(data) + "_me/" , 'me']
        exec(a_script)
        script_descriptor.close()
