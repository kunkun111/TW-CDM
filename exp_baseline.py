# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:45:13 2023

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
import time
import arff

from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




path = '/home/kunwang/Data/work5/data/'

# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])

chunkszie = 100


 
# main part of model
datasetname1 = 's1'
data1 = load_arff(path, datasetname1)
data1 = data1.values

datasetname2 = 's6'
data2 = load_arff(path, datasetname2)
data2 = data2.values


# initial the training sets of two streams
x1_train = data1[0:chunkszie, :-1]
y1_train = data1[0:chunkszie, -1]

x2_train = data2[0:chunkszie, :-1]
y2_train = data2[0:chunkszie, -1]

'''
# realworld
datasetname = 'weather'
# datasetname = 'elecNorm'

data = load_arff(path, datasetname)
data = data.values
size = int(data.shape[0]/2)

data1 = data[0:size, :]
data2 = data[size:, :]

x1_train = data1[0:chunkszie, :-1]
y1_train = data1[0:chunkszie, -1]

x2_train = data2[0:chunkszie, :-1]
y2_train = data2[0:chunkszie, -1]
'''


x_train_new = np.vstack((x1_train, x2_train))
y_train_new = np.hstack((y1_train, y2_train))
    
    
# Begin train the learning model
model_learn = GradientBoostingRegressor(random_state=0)
model_learn.fit(x_train_new, y_train_new)
    
    
# prepare for the testing process
accuracy1_total = []
accuracy2_total = []

fscore1_total = []
fscore2_total = []

data1_label = np.zeros(chunkszie)
data2_label = np.ones(chunkszie)


# model testing and adaptation
for i in range(chunkszie, data1.shape[0], chunkszie):
    
    x1_test = data1[i:i+chunkszie, :-1]
    y1_test = data1[i:i+chunkszie, -1]

    x2_test = data2[i:i+chunkszie, :-1]
    y2_test = data2[i:i+chunkszie, -1]
    
    
    # test the model on two data streams
    pred1_test = model_learn.predict(x1_test)
    pred2_test = model_learn.predict(x2_test)
    
    
    # get the label of test results of two data streams
    pred1_label = (pred1_test >= 0.5)
    pred2_label = (pred2_test >= 0.5)
    
    
    # calculate the accuracy of two results
    accuracy1_test = metrics.accuracy_score(y1_test, pred1_label)
    accuracy2_test = metrics.accuracy_score(y2_test, pred2_label)
    
    
    accuracy1_total.append(accuracy1_test)
    accuracy2_total.append(accuracy2_test)
    
    
    fscore1_test = metrics.f1_score(y1_test, pred1_label)
    fscore2_test = metrics.f1_score(y2_test, pred2_label)
    
    
    fscore1_total.append(fscore1_test)
    fscore2_total.append(fscore2_test)


print('S1 accuracy:', np.average(accuracy1_total))
print('S2 accuracy:', np.average(accuracy2_total))

print('S1 f1:', np.average(fscore1_total))
print('S2 f1:', np.average(fscore2_total))





