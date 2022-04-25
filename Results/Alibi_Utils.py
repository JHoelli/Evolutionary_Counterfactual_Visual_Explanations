import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def calculateData(path):
    data=pickle.load(open(path,'rb'))
    #print(explenation)
    orig_output = data['orig_proba']
    print(len(data['all']))
    n_cfs = np.array([len(data['all'][iter_cf]) for iter_cf in range(10)])
    examples = {}
    for ix, n in enumerate(n_cfs):
        if n > 0:
            examples[ix] = {'ix': ix, 'lambda': data['all'][ix][0]['lambda'],
                            'X': data['all'][ix][0]['X'], 'output': data['all'][ix][0]['proba'][0][data['all'][ix][0]['class']],'distance':data['all'][ix][0]['distance']}

    sparsity=[]
    distance=[]
    output=[]
    for i, key in enumerate(examples.keys()):
        distance.append(examples[key]['distance']/784)
        output.append(1-(np.abs(examples[key]['output']-(1- data['orig_proba']))))
        substract = abs(np.subtract(np.load('./Alibi_0/comp.npy').reshape(-1)*784,examples[key]['X'].reshape(-1)*784 ))
        print(substract)
        substract[substract < 20] = 0
        sp = np.count_nonzero(substract)
        print(sp)
        sparsity.append(sp/784)
    return distance, sparsity,output


def last (path,orgim):
    data = pickle.load(open(path, 'rb'))
    last=data['all'][len(data['all'])-1]
    #n_cfs = np.array([len(data['all'][len(data['all'])-1][iter_cf]) for iter_cf in range(10)])
    examples = {}
    for ix, n in enumerate(last):
        if ix > 0:
            examples[ix] = {'ix': ix, 'lambda': data['all'][len(data['all'])-1][ix]['lambda'],
                            'X': data['all'][len(data['all'])-1][ix]['X'],
                            'output': data['all'][len(data['all'])-1][ix]['proba'][0][data['all'][len(data['all'])-1][ix]['class']],
                            'distance': data['all'][len(data['all'])-1][ix]['distance']}
    sparsity = []
    distance = []
    output = []
    for i, key in enumerate(examples.keys()):
        dis= np.mean(abs(
            np.subtract(pickle.load(open(orgim,'rb')).reshape(-1), examples[key]['X'].reshape(-1) *255)))
        distance.append(examples[key]['distance'] / 255)
        output.append(1 - (examples[key]['output']))
        substract = abs(
            np.subtract(pickle.load(open(orgim,'rb')).reshape(-1), examples[key]['X'].reshape(-1) *255))
        substract[substract < 20] = 0
        sp = np.count_nonzero(substract)
        sparsity.append(sp / 784)
    return distance, sparsity, output

def counterfactualonly (path,orgim):
    sparsity = []
    distance = []
    output = []
    for a in os.listdir(path):
        examples = pickle.load(open(path+'/'+a+'/explain_Target.pkl', 'rb'))
        #try:
        #    distance.append(examples['cf']['distance'] / 784)
        #except:
            #print(examples['cf']['X'].reshape(-1))
        dis=np.mean(np.abs(np.subtract(pickle.load(open(orgim,'rb')).reshape(-1)/255, examples['cf']['X'].reshape(-1))))
        distance.append(dis)
        #print(examples['cf']['proba'])
       # print(examples['orig_proba'])
        pred_class = examples.cf['class']
        proba = examples.cf['proba'][0][pred_class]
        #print(examples.cf['proba'][0])
        #print(proba)
        #print(examples['orig_proba'])
        #output.append(1 - (np.abs(proba - (1-examples['orig_proba']))))
        output.append(1 - proba)

        substract = abs(
            np.subtract(pickle.load(open(orgim,'rb')).reshape(-1), examples['cf']['X'].reshape(-1) * 255))
        #print(substract)
        substract[substract < 20] = 0
        sp = np.count_nonzero(substract)
        #print(sp)
        sparsity.append(sp / 784)
   # print(distance)
    return distance, sparsity, output