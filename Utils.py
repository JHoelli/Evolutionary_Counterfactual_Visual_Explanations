from deap.tools.emo import selNSGA2
from deap import creator
from deap import base
import os
import pickle
import re
import numpy as np

def getFitnessFromPareto(ParetoFront):
    distance= []
    sparsity=[]
    output=[]
    for a in ParetoFront:
        print(a.fitness)
        distance.append(a.fitness.values[0])
        sparsity.append(a.fitness.values[1])
        output.append(a.fitness.values[2])
        #for b in a.fitness.values:
        #    print(b)
    return distance,sparsity,output

def getOverallBestPics(path, size=1):
    for picnumber in os.listdir(path):
        all = []
        """Has to be always there for pareto Front"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        if not picnumber.endswith('.txt'):
            for file in os.listdir(path+'/'+picnumber):
                if file.startswith('pareto'):
                    infile = open(path + '/'+picnumber+'/' + file, 'rb')
                    new_dict = pickle.load(infile)
                    infile.close()
                    for a in new_dict:
                        all.append(a)
            print(len(all))
            if size!= None:
                res = selNSGA2(all,size)
            print(len(res))
            pickle.dump(res, open(path+'/'+str(picnumber) + '/Best_'+str(size)+'_Images.pkl', "wb"), -1)


def getOverallBestPicsOutput(path, size=1):
    for picnumber in os.listdir(path):
        all = []
        """Has to be always there for pareto Front"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        if not picnumber.endswith('.txt'):
            max=0
            for file in os.listdir(path+'/'+picnumber):
                if file.startswith('pareto'):
                    search = re.search(r'\d', file)
                    infile = open(path + '/'+picnumber+'/' + file, 'rb')
                    new_dict = pickle.load(infile)
                    infile.close()
                    for a in new_dict:
                        if np.argmax(a.output) == int(search[0]):
                            # print('if')
                            c = a.output[0][np.argmax(a.output)]
                            if c > max:
                                all.append(a)
            print(len(all))
            pickle.dump(all, open(path+'/'+str(picnumber) + '/Best_Output_Overall'+'_Images.pkl', "wb"), -1)

def getOverallBestPicsDistance(path, size=1):
    for picnumber in os.listdir(path):
        all = []
        """Has to be always there for pareto Front"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        if not picnumber.endswith('.txt'):
            max=0
            for file in os.listdir(path+'/'+picnumber):
                if file.startswith('pareto'):
                    search = re.search(r'\d', file)
                    infile = open(path + '/'+picnumber+'/' + file, 'rb')
                    new_dict = pickle.load(infile)
                    infile.close()
                    for a in new_dict:
                        c=a.fitness.values[0]
                        if c > max:
                            all.append(a)
            print(len(all))
            pickle.dump(all, open(path+'/'+str(picnumber) + '/Best_Distance_Overall'+'_Images.pkl', "wb"), -1)

def getIslandBestPic(path, size=1):
    '''For each Island get the Best Image (fitnesswise), that fullfills the counterfactual constraint'''
    print(path)
    for pic in os.listdir(path):
        #creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        #creator.create("Individual", list, fitness=creator.FitnessMin)
        if pic.startswith('pareto'):
            list=[]
            search = re.search(r'\d', pic)
            #print('Search for ',search[0])
            dict=pickle.load(open(path+'/'+pic,'rb'))
            for b in dict:
                #print(b.output)
                #print(b.fitness.values)
                if np.argmax(b.output) == int(search[0]):
                    #print('Pic ',np.argmax(b.output))
                    #print('Org', search[0])
                    #print('if hit')
                    list.append(b)
                    #print(len(list))
            res= selNSGA2(list,size)
            pickle.dump(res, open(path +'/Best_Island_'+str(search[0])+'_'+ str(size) + '_Images.pkl', "wb"), -1)


def getIslandBestPicOutput(path, size=1):
    '''For each Island get the Best Image (Output Probability), that fullfills the counterfactual constraint'''
    #print(path)
    for pic in os.listdir(path):
        if pic.startswith('pareto'):
            search = re.search(r'\d', pic)
            dict=pickle.load(open(path+'/'+pic,'rb'))
            max=0
            for b in dict:
                if np.argmax(b.output) == int(search[0]):
                    #print('if')
                    a=b.output[0][np.argmax(b.output)]
                    if a>max:
                        max=a
                        res=[b]
            pickle.dump(res, open(path +'/Output_Best_Island_'+str(search[0])+'_'+ str(size) + '_Images.pkl', "wb"), -1)
            #print(res)
            #print('saved to  ',path +'/Output_Best_Island_'+str(search[0])+'_'+ str(size) + '_Images.pkl' )


