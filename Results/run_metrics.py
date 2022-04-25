import pandas as pd
import numpy as np
from metrics import yNN,get_distances,redundancy,validity
import pickle
import os
from deap import base
from deap import creator
import Additional_Help_Functions as helper
import argparse

class Benchmark:

    def __init__(
        self,
        mlmodel,
        counterfactuals,
        factuals,y,target,data='MNIST'
    ) -> None:

        self._mlmodel = mlmodel
        self._counterfactuals = counterfactuals

        self._factual = factuals.copy()
        self.y = y
        self.target= target
        self.data=data

    def compute_ynn(self) -> pd.DataFrame:
        """
        Computes y-Nearest-Neighbours for generated counterfactuals
        Returns
        -------
        pd.DataFrame
        counterfactuals,
    mlmodel,
    y:
        """
        ynn = yNN(
               self._counterfactuals,  self._mlmodel,self.data, self.y
            )
        columns = ["y-Nearest-Neighbours"]

        return pd.DataFrame(ynn, columns=columns)

    def compute_distances(self) -> pd.DataFrame:
        """
        Calculates the distance measure and returns it as dataframe
        Returns a List of differences ?
        -------
        pd.DataFrame
        """
        columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]


        arr_f = self._factual#.to_numpy()
        arr_cf = self._counterfactuals#.to_numpy()
        distances=[]
        for cf in arr_cf:
            #print(type(cf))
            #print(arr_f.shape)
            #print(cf.shape)
            if type(cf)==dict:
                distance = get_distances(arr_f.reshape(1, -1), np.array([cf['X'].reshape(-1)]))
            else:
                distance = get_distances(arr_f.reshape(1,-1), np.array([cf]))
            distances.append(distance[0])

        output = pd.DataFrame(distances, columns=columns)
        return output


    def compute_redundancy(self) -> pd.DataFrame:
        """
        Computes redundancy for each counterfactual
        Returns
        -------
        pd.Dataframe
        """
        redundancies = redundancy(
                self._factual, self._counterfactuals, self._mlmodel
            )

        columns = ["Redundancy"]

        return pd.DataFrame(redundancies, columns=columns)

    def compute_success_rate(self) -> pd.DataFrame:
        """
        Computes success rate for the whole recourse method.
        Returns
        -------
        pd.Dataframe
        """
        #TODO Return 1 or 0
        if(type(self._counterfactuals)!= dict):
            rate = validity(self._counterfactuals, self.target,self._mlmodel)
        else:
            pass
        columns = ["Success_Rate"]

        return pd.DataFrame(rate, columns=columns)

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.
        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            self.compute_distances(),
            self.compute_redundancy(),
            self.compute_ynn(),
            #self.compute_success_rate(),
        ]

        output = pd.concat(pipeline, axis=1)

        return output
target = 0
if __name__ == '__main__':
    '''
    This is for our Code! 

    '''
    import tensorflow as tf
    import re
    import pickle5 as pickle

    parser = argparse.ArgumentParser(description='Dataset.')
    parser.add_argument('dataset', metavar='d', type=str, nargs=1, help='MNIST or Fashion')
    args = parser.parse_args()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    # counterfactuals=[]
    print(args.dataset)
    if args.dataset[0] =='MNIST':
        data='mnist'
        ml_model = tf.keras.models.load_model('../Model/cnn_MNIST.h5')
    elif args.dataset[0] == 'Fashion':
        ml_model = tf.keras.models.load_model('../Model/cnn_Fashion_mnist.h5')
        data='fashion'

    y = 5

    path = f'./Benchmark/Evo/{data}_me/'
    outputs = []

    for image_number in os.listdir(path):
        # print(type(os.listdir(f'{path}/{image_number}')))
        if not image_number.endswith('.txt') and not image_number.endswith('.csv'):
            counterfactuals = []
            target = []
            for a in sorted(os.listdir(f'{path}/{image_number}')):
                if not a.endswith('.txt'):
                    print(a)
                    if a.startswith('beginning'):
                        original = pickle.load(open(f'{path}/{image_number}/{a}', 'rb'))
                        print('Shape of Original', original.shape)
                    if a.startswith('BestHof'):
                        cf = np.array(pickle.load(open(f'./{path}/{image_number}/{a}', 'rb')))
                        print(type(cf[0]))
                        target = target + [int(re.findall(r"\d", a)[0])]
                        c = helper.reshape_to_image(np.array(cf[0]), (28, 28, 1)).reshape(-1)
                        ci = creator.Individual(c)
                        ci.output = ml_model.predict(c.reshape(1, 28, 28, 1) / 155)
                        # c.fitness = cf[0].fitness
                        counterfactuals = counterfactuals + [ci]  # Former: counterfactuals +
                        print('Counterfatuals',np.array(counterfactuals).shape)
            benchmark = Benchmark(ml_model, counterfactuals, original, y, target,data=args.dataset[0])
            output = benchmark.run_benchmark()
            if len(outputs) <= 1:
                print('if')
                outputs = output
            else:
                print('ELSE')
                outputs = pd.concat([outputs, output], ignore_index=True)
    outputs.to_csv(f'{path}/benchmark_results.csv')

    '''
        Code for Alibi! 
    '''
    org_path = f'./Benchmark/Evo/{data}_me'
    paths = [f'./Alibi/VanLooveren/{args.dataset[0]}/Classwise/', f'./Alibi/Wachter/{args.dataset[0]}/Classwise/']


    outputs=[]

    for path in paths:
        for image_number in  os.listdir(path):

            for file in os.listdir(f'{org_path}/{image_number}'):
                if file.startswith('beginning'):
                    original = pickle.load(open(f'./{org_path}/{image_number}/{file}', 'rb'))
                    print('Shape of Original', original.shape)
            print(image_number)
            counterfactuals = []
            for a in os.listdir(f'{path}/{image_number}'):

                #length = len(os.listdir(f'{path}/{image_number}/{a}'))
                if os.listdir(f'{path}/{image_number}/{a}'):
                    file = open(f'{path}/{image_number}/{a}/explain_Target.pkl', 'rb')
                    counterfactuals = counterfactuals + [pickle.load(file).cf]
                    file.close()
                    y=2 #' nb_neighbors < nbsamples'
                    benchmark= Benchmark(ml_model,np.array(counterfactuals), original,y, f'{path}/{image_number}/{a}', data=args.dataset[0])
                    output=benchmark.run_benchmark()
                    #output['success']=[1]
                    print(len(outputs))
                    if len(outputs)<=1:
                        print('if')
                        outputs=output
                    else:
                        print('ELSE')
                        outputs = pd.concat([outputs, output], ignore_index=True)
            #outputs=outputs.append(output, ignore_index=True)

        outputs.to_csv(f'{path}/benchmark_results.csv')


