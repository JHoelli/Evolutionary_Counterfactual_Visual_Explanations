import os

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from typing import List
from deap import base
from deap import creator

import Additional_Help_Functions as helper

def validity (counterfactuals, target,ml_model):
    nums=[]
    if type(target)!=str:
        #length= len(counterfactuals)

        for tar, cf in zip(target,counterfactuals):
            label = np.argmax(ml_model.predict(np.array(cf).reshape(1,28,28,1)/255))
            print('tar',tar)
            print('label',label)
            if label == tar:
                nums.append([1])#label[label==target]
            else:
                nums.append([0])
    return np.array(nums)

def get_distances(factual: np.ndarray, counterfactual: np.ndarray) -> List[List[float]]:
    """
    Computes distances 1 to 4.
    All features have to be in the same order (without target label).
    Parameters
    ----------
    factual: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    counterfactual: np.ndarray
        Normalized and encoded array with counterfactual data
        Shape: NxM
    Returns
    -------
    list: distances 1 to 4
    """
    factual.reshape(1,-1)
    if factual.shape != counterfactual.shape:
        raise ValueError("Shapes of factual and counterfactual have to be the same")
    if len(factual.shape) != 2:
        raise ValueError(
            "Shapes of factual and counterfactual have to be 2-dimensional"
        )

    # get difference between original and counterfactual
    delta = get_delta(factual, counterfactual)
    d= counterfactual.shape[1] #input dimension
    d1 = d1_distance(delta)
    d2 = d2_distance(delta)
    d3 = d3_distance(delta)
    d4 = d4_distance(delta)
    return [[d1[i]/d, d2[i]/d, d3[i]/d, d4[i]/d] for i in range(len(d1))]


def d1_distance(delta: np.ndarray) -> List[float]:
    """
    Computes D1 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    # compute elements which are greater than 0
    return np.sum(delta != 0, axis=1, dtype=np.float).tolist()


def d2_distance(delta: np.ndarray) -> List[float]:
    """
    Computes D2 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """

    return np.sum(np.abs(delta), axis=1, dtype=np.float).tolist()


def d3_distance(delta: np.ndarray) -> List[float]:
    """
    Computes D3 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    return np.sum(np.square(np.abs(delta)), axis=1, dtype=np.float).tolist()


def d4_distance(delta: np.ndarray) -> List[float]:
    """
    Computes D4 distance
    Parameters
    ----------
    delta: np.ndarray
        Difference between factual and counterfactual
    Returns
    -------
    List[float]
    """
    return np.max(np.abs(delta), axis=1).tolist()


def get_delta(instance: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """
    Compute difference between original instance and counterfactual
    Parameters
    ----------
    instance: np.ndarray
        Normalized and encoded array with factual data.
        Shape: NxM
    cf: : np.ndarray
        Normalized and encoded array with counterfactual data.
        Shape: NxM
    Divided by 255 for normalization
    Returns
    -------
    np.ndarray
    """
    return (cf - instance)/255


def yNN(
    counterfactuals,
    mlmodel,data,
    y: int,
) -> List[float]:
    """
    Parameters
    ----------
    counterfactuals: Generated counterfactual examples
    recourse_method: Method we want to benchmark
    y: Number of
    Returns
    -------
    float
    """
    number_of_diff_labels = 0
    if data=='MNIST':
        x_train,y_train,_,_=helper.load_mnist()
    elif data == 'Fashion':
        x_train, y_train, _, _ = helper.load_fashion_mnist()
    x_train = x_train.reshape(-1, 784)
    if type(counterfactuals[0]) == dict:
        labels = [np.argmax(cf['proba']) for cf in counterfactuals]
        counterfactuals=[cf['X'].reshape(-1) for cf in counterfactuals]
    else:
        try:
            labels = [np.argmax(cf.output) for cf in counterfactuals]
        except:
            labels=[np.argmax(mlmodel.predict(cf.reshape(1,28,28,1)/255)) for cf in counterfactuals]
        counterfactuals = [np.array(cf) for cf in counterfactuals]
    N = np.array(counterfactuals).shape[0]
    data = np.concatenate( (x_train,counterfactuals))
    nbrs = NearestNeighbors(n_neighbors=y).fit(np.array(data))

    counterfactuals=pd.DataFrame(counterfactuals)
    calc=[]
    for i, row in  counterfactuals.iterrows():
        knn = nbrs.kneighbors(row.values.reshape((1, -1)), y, return_distance=False)[0]
        cf_label = labels[i] #row[mlmodel.data.target]

        for idx in knn:
            neighbour = data[idx] #counterfactuals.iloc[idx]
            neighbour = neighbour.reshape((1, -1))
            neighbour_label = np.argmax(mlmodel.predict(neighbour.reshape(1,28,28,1)/255))
            if not cf_label == neighbour_label:
                number_of_diff_labels += 1
        calc.append([1 - (1 / (N * y)) * number_of_diff_labels])

    return np.array(calc)

def redundancy(original, counterfactuals, mlmodel) :
    """
    Computes Redundancy measure for every counterfactual
    Parameters
    ----------
    factuals: Encoded and normalized factual samples
    counterfactuals: Encoded and normalized counterfactual samples
    mlmodel: Black-box-model we want to discover
    Returns
    -------
    List with redundancy values per counterfactual sample
    """
    #df_enc_norm_fact = factuals.reset_index(drop=True)
    #df_cfs = counterfactuals.reset_index(drop=True)

    if type(counterfactuals[0])==dict:
        #print(counterfactuals[0])
        labels = [np.argmax(cf['proba']) for cf in counterfactuals]
    else:
        labels = [np.argmax(cf.output) for cf in counterfactuals]
    #print(labels)

    df_cfs = np.array(counterfactuals)
    redun=[]
    for i in range (0,len(df_cfs)):
        #print(np.array(df_cfs[i]))
        if type(df_cfs[i])==dict:
            redun.append(compute_redundancy(original, np.array(df_cfs[i]['X']), mlmodel, labels[i]))
        else:
            redun.append(compute_redundancy(original,np.array(df_cfs[i]),mlmodel,labels[i]))
    #df_cfs["redundancy"] = df_cfs.apply(
    #    lambda x: compute_redundancy(
    #        original , x.values, mlmodel, labels.iloc[x.name]
    #    ),
    #    axis=1,
    #)
    #print(redun)
    return redun#df_cfs["redundancy"].values.reshape((-1, 1)).tolist()


def compute_redundancy(
    fact: np.ndarray, cf: np.ndarray, mlmodel, label_value: int
) -> int:
    red = 0
    #print(fact)
    shape= fact.shape
    #print(shape)
    fact=fact.reshape(-1)
    cf=cf.reshape(shape[0]*shape[1]*shape[2])
    for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape
        print(fact[col_idx] )
        print( cf[col_idx])
        if fact[col_idx] != cf[col_idx]:
            temp_cf = np.copy(cf)

            temp_cf[col_idx] = fact[col_idx]

            temp_pred = np.argmax(mlmodel(temp_cf.reshape((1,shape[0],shape[1],shape[2]))))

            if temp_pred == label_value:
                red += 1

    return red