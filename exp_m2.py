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


# sample selection 
def segment1(path):    

    # transfer list to array 
    list_array = np.array(path)
    
    # get the duplicate data
    data_total = np.unique(list_array[:, 1])
    
    # get the duplicate data from list
    list2 = list_array[:, 1].tolist()
    data_count = []
    for i in data_total:
        if list2.count(i)>1:
            data_count.append(i)
    
    # get the index of duplicate data
    segment_list = []
    for i in range(len(data_count)):
        index_list = []
        for j in range(len(list2)):
            if list2[j] == data_count[i]:
                index_list.append(j) 
        segment_list.extend(index_list)
    
    return data_count, segment_list


# sample selection 
def segment2(path):    

    # transfer list to array 
    list_array = np.array(path)
    
    # get the duplicate data
    data_total = np.unique(list_array[:, 0])
    
    # get the duplicate data from list
    list2 = list_array[:, 0].tolist()
    data_count = []
    for i in data_total:
        if list2.count(i)>1:
            data_count.append(i)
    
    # get the index of duplicate data
    segment_list = []
    for i in range(len(data_count)):
        index_list = []
        for j in range(len(list2)):
            if list2[j] == data_count[i]:
                index_list.append(j) 
        segment_list.extend(index_list)
    
    return data_count, segment_list


# drift region identification 
def region(path, data_count_drift):    
    
    # transfer list to array 
    list_array = np.array(path)
    
    # get the duplicate data
    data_total = np.unique(list_array[:, 0])
    
    # get the duplicate data from list
    list2 = list_array[:, 0].tolist()
    
    # get the index of duplicate data
    drift_list = []
    for i in range(len(data_count_drift)):
        index_list = []
        for j in range(len(list2)):
            if list2[j] == data_count_drift[i]:
                index_list.append(j) 
        drift_list.extend(index_list)
    
    return drift_list



# def draw_region(x, y, drift_region):
    
#     label1_idx = np.array(np.where(y == 0)).flatten()
#     label2_idx = np.array(np.where(y == 1)).flatten()
    
#     # label1 data prepare
#     data1 = x[label1_idx,:]
#     x1 = data1[:, 0]
#     y1 = data1[:, 1]
    
#     # label2 data prepare
#     data2 = x[label2_idx]
#     x2 = data2[:, 0]
#     y2 = data2[:, 1]
    
#     # label3 data prepare
#     data3 = drift_region
#     x3 = data3[:, 0]
#     y3 = data3[:, 1]

#     # draw figure
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.scatter(y1, x1, c='red', label='data1', marker='.', alpha=1)
#     ax.scatter(y2, x2, c='blue', label='data2', marker='.', alpha=1)
#     ax.scatter(y3, x3, c='lime', label='drift region', marker='.', alpha=1)
    
#     ax.set_xlabel('x1', fontdict={'size': 10, 'color': 'black'})
#     ax.set_ylabel('x2', fontdict={'size': 10, 'color': 'black'})
    
#     ax.legend(loc = 'best')
#     plt.show()
    
    
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

# initial train a model and get the error rate
clf = GradientBoostingClassifier(random_state=0)
model = clf.fit(x_train_new, y_train_new)
pred1 = model.predict(x1_train)
pred2 = model.predict(x2_train)


# calculate the error
eva1 = np.zeros(pred1.shape[0])
right = 0
for i in range(1, pred1.shape[0]):
    accuracy1 = metrics.accuracy_score(y1_train[0:i], pred1[0:i])
    eva1[i] = accuracy1
    
eva2 = np.zeros(pred2.shape[0])
right = 0
for i in range(1, pred2.shape[0]):
    accuracy2 = metrics.accuracy_score(y2_train[0:i], pred2[0:i])
    eva2[i] = accuracy2
    
    
# dynamic time warping
#----------------------------------
s1 = eva1
s2 = eva2

path = dtw.warping_path(s1, s2)
# print(path)
# fig, axes = dtwvis.plot_warping(s1, s2, path)
# axes[0].set_xlabel('Time stamp')
# axes[0].set_ylabel('Error rate')
# axes[1].set_xlabel('Time stamp')
# axes[1].set_ylabel('Error rate') 


# sample selection
path_array = np.array(path)
data_count1, index_del1 = segment1(path)
data_count2, index_del2 = segment2(path)


if len(index_del1) > 0 and len(index_del2) > 0:               

    # find the segment index of stream 1,2   
    data1_idx = path_array[index_del1, 0]
    data2_idx = path_array[index_del2, 1]
    
    
    # (for Boosting model learning) update the total train set by combine updated train set of stream 1,2
    x_train_new = np.vstack((x1_train, x2_train))
    y_train_new = np.hstack((y1_train, y2_train))
    
    # (for separated Tree model learning) get the seprate train set of stream 1,2 based on segment data
    x_train_new_s1 = x1_train[data1_idx, :]
    y_train_new_s1 = y1_train[data1_idx]
    
    x_train_new_s2 = x2_train[data2_idx, :]
    y_train_new_s2 = y2_train[data2_idx]
    
    # Begin train the learning model
    model_learn = GradientBoostingRegressor(random_state=0)
    model_learn.fit(x_train_new, y_train_new)
    
    s1_model = DecisionTreeRegressor(random_state=0)
    s1_model.fit(x_train_new_s1, y_train_new_s1)
    
    s2_model = DecisionTreeRegressor(random_state=0)
    s2_model.fit(x_train_new_s2, y_train_new_s2)
    
else:
    
    x_train_new = np.vstack((x1_train, x2_train))
    y_train_new = np.hstack((y1_train, y2_train))
    
    # Begin train the learning model
    model_learn = GradientBoostingRegressor(random_state=0)
    model_learn.fit(x_train_new, y_train_new)
    
    s1_model = DecisionTreeRegressor(random_state=0)
    s1_model.fit(x_train_new, y_train_new)
    
    s2_model = DecisionTreeRegressor(random_state=0)
    s2_model.fit(x_train_new, y_train_new)
    

# prepare for the testing process
accuracy1_total = []
accuracy2_total = []


fscore1_total = []
fscore2_total = []


data1_label = np.zeros(chunkszie)
data2_label = np.ones(chunkszie)


# initial accuracy
accuracy1_ini = accuracy1
accuracy2_ini = accuracy2


# test set initial
x_test_ini = x_train_new
y_test_ini = y_train_new


# total drift frequency

total_frequency = 0


# model testing and adaptation
for i in range(chunkszie, data1.shape[0], chunkszie):
    
    x1_test = data1[i:i+chunkszie, :-1]
    y1_test = data1[i:i+chunkszie, -1]

    x2_test = data2[i:i+chunkszie, :-1]
    y2_test = data2[i:i+chunkszie, -1]
    
    
    # test the model on two data streams
    pred1_test = model_learn.predict(x1_test)
    pred2_test = model_learn.predict(x2_test)
    
    
    # test tree models on two data streams
    pred1_test_tree = s1_model.predict(x1_test)
    pred2_test_tree = s1_model.predict(x2_test)
    
    
    # get the test of two data streams
    pred1_test_total = pred1_test + 0.1*pred1_test_tree
    pred2_test_total = pred2_test + 0.1*pred2_test_tree
    
    # pred1_test_total = pred1_test
    # pred2_test_total = pred2_test
    
    
    # get the label of test results of two data streams
    pred1_label = (pred1_test_total >= 0.5)
    pred2_label = (pred2_test_total >= 0.5)
    
    
    # calculate the accuracy of two results
    accuracy1_test = metrics.accuracy_score(y1_test, pred1_label)
    accuracy2_test = metrics.accuracy_score(y2_test, pred2_label)
    
    accuracy1_total.append(accuracy1_test)
    accuracy2_total.append(accuracy2_test)
    
    
    fscore1_test = metrics.f1_score(y1_test, pred1_label)
    fscore2_test = metrics.f1_score(y2_test, pred2_label)
    
    
    fscore1_total.append(fscore1_test)
    fscore2_total.append(fscore2_test)
    
    
    # calculate the error
    eva1 = np.zeros(pred1_test.shape[0])
    right = 0
    for a in range(1, pred1_test.shape[0]):
        accuracy1 = metrics.accuracy_score(y1_test[0:a], pred1_label[0:a])
        eva1[a] = accuracy1
        
    eva2 = np.zeros(pred2_test.shape[0])
    right = 0
    for b in range(1, pred2_test.shape[0]):
        accuracy2 = metrics.accuracy_score(y2_test[0:b], pred2_label[0:b])
        eva2[b] = accuracy2

        
    # dynamic time warping
    s1 = eva1
    s2 = eva2

    path = dtw.warping_path(s1, s2)
    # print(path)
    # fig, axes = dtwvis.plot_warping(s1, s2, path)
    # axes[0].set_xlabel('Time stamp')
    # axes[0].set_ylabel('Error rate')
    # axes[1].set_xlabel('Time stamp')
    # axes[1].set_ylabel('Error rate') 


    # sample selection
    path_array = np.array(path)
    data_count1, index_del1 = segment1(path)
    data_count2, index_del2 = segment2(path)
    
    
    # drift detection
    drift_fre = 0
    data_count_drift = []
    for idx in range (1, len(data_count2)):
        if s1[data_count2[idx-1]] > s1[data_count2[idx]]:
            drift_fre += 1
            data_count_drift.append(data_count2[idx])
            
    # print('Drift frequency:', drift_fre)
    # total_frequency = total_frequency + drift_fre
    # print('Data count drift:', data_count_drift )
    
    
    # identify drift region
    drift_list = region(path, data_count_drift)
    # print('Drift List:', drift_list)
    
    
    # data augmentation
    if len(index_del1) > 0 and len(index_del2) > 0: 
           
        # find the segment index of stream 1,2   
        data1_idx = path_array[index_del1, 0]
        data2_idx = path_array[index_del2, 1]
        
        # drift detection
        data2_drift_idx = path_array[drift_list, 1]
        data2_drift_x = x2_test[data2_drift_idx, :]
        data2_drift_y = y2_test[data2_drift_idx]
        # draw_region(x2_test, y2_test, data2_drift_x) 
        
        # (for Boosting model learning) update the total train set by combine updated train set of stream 1,2
        x_test_new = np.vstack((x1_test, x2_test))
        y_test_new = np.hstack((y1_test, y2_test))
        
        x_test_new_s1 = x1_test[data1_idx, :]
        y_test_new_s1 = y1_test[data1_idx]
        
        x_test_new_s2 = x2_test[data2_idx, :]
        y_test_new_s2 = y2_test[data2_idx]
        
            
        # retrain the learning model
        model_learn = GradientBoostingRegressor(random_state=0)
        model_learn.fit(x_test_new, y_test_new)
        
        s1_model = DecisionTreeRegressor(random_state=0)
        s1_model.fit(x_test_new_s1, y_test_new_s1)
    
        s2_model = DecisionTreeRegressor(random_state=0)
        s2_model.fit(x_test_new_s2, y_test_new_s2)
            
            
    else:
        x_test_new = np.vstack((x1_test, x2_test))
        y_test_new = np.hstack((y1_test, y2_test))
        
        # retrain the learning model
        model_learn = GradientBoostingRegressor(random_state=0)
        model_learn.fit(x_test_new, y_test_new)
    
        s1_model = DecisionTreeRegressor(random_state=0)
        s1_model.fit(x_test_new, y_test_new)
    
        s2_model = DecisionTreeRegressor(random_state=0)
        s2_model.fit(x_test_new, y_test_new)



print('S1 accuracy:', np.average(accuracy1_total))
print('S2 accuracy:', np.average(accuracy2_total))


print('S1 f1:', np.average(fscore1_total))
print('S2 f1:', np.average(fscore2_total))





