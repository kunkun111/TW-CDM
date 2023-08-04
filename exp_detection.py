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
from tqdm import tqdm
import time
import arff

from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



path = '/home/kunwang/Data/work5/dataset/Synthetic/'


# load .arff dataset
def load_arff(path, dataset_name, seeds):
    file_path = path + dataset_name + '/'+ dataset_name + str(seeds) + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


# sample selection 
def segment(path):    

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



def draw_region(x, y, drift_region):
    
    label1_idx = np.array(np.where(y == 0)).flatten()
    label2_idx = np.array(np.where(y == 1)).flatten()
    
    # label1 data prepare
    data1 = x[label1_idx,:]
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    z1 = data1[:, 2]
    
    # label2 data prepare
    data2 = x[label2_idx]
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    z2 = data2[:, 2]
    
    # label3 data prepare
    data3 = drift_region
    x3 = data3[:, 0]
    y3 = data3[:, 1]
    z3 = data3[:, 2]

    # draw figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(y1, x1, z1, c='red', label='data1', marker='.', alpha=1)
    ax.scatter(y2, x2, z2, c='blue', label='data2', marker='.', alpha=1)
    ax.scatter(y3, x3, z3, c='lime', label='drift region', marker='.', alpha=1)
    
    ax.set_xlabel('x1', fontdict={'size': 10, 'color': 'black'})
    ax.set_ylabel('x2', fontdict={'size': 10, 'color': 'black'})
    ax.set_zlabel('Noise', fontdict={'size': 10, 'color': 'black'})
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 15)
    
    ax.legend(loc = 'best')

    plt.show()
    
    
    
# main part of model
datasetname = 'SEAa'

seed1 = 0
data = load_arff(path, datasetname, seed1)
data = data.values


# seed2 = 1
# data2 = load_arff(path, datasetname, seed2)
# data2 = data2.values


# data prepare
data1 = data[0:5000, :]
data2 = data[5000:10000, :]


# initial the training sets of two streams
x1_train = data1[0:100, :-1]
y1_train = data1[0:100, -1]

x2_train = data2[0:100, :-1]
y2_train = data2[0:100, -1]


# initial train a model and get the error rate
clf = GradientBoostingClassifier(random_state=0)
model = clf.fit(x1_train, y1_train)
pred1 = model.predict(x1_train)
pred2 = model.predict(x2_train)


# calculate the error
eva1 = np.zeros(pred1.shape[0])
right = 0
for i in range(1, pred1.shape[0]):
    accuracy1 = metrics.accuracy_score(y1_train[0:i], pred1[0:i])
    eva1[i] = accuracy1
    
    # if pred1[i] == y1_train[i]:
    #     right = 1
    #     eva1[i] = right 
    # else:
    #     right = 0
    #     eva1[i] = right
    
eva2 = np.zeros(pred2.shape[0])
right = 0
for i in range(1, pred2.shape[0]):
    accuracy2 = metrics.accuracy_score(y2_train[0:i], pred2[0:i])
    eva2[i] = accuracy2
    
    # if pred2[i] == y2_train[i]:
    #     right = 1
    #     eva2[i] = right 
    # else:
    #     right = 0
    #     eva2[i] = right
    
    
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
data_count, index_del = segment(path)
# print(index_del)
 
             
if len(index_del) > 0:               

    data2_idx = path_array[index_del, 1]
    # data1_idx = np.delete(path_array[:, 0], np.array(index_del2)) 
    
    x_train_new_s1 = np.delete(x1_train, data2_idx)
    y_train_new_s1 = np.delete(y1_train, data2_idx)
    
    x_train_new = np.vstack((x1_train, x2_train[data2_idx, :]))
    y_train_new = np.hstack((y1_train, y2_train[data2_idx]))
    
    # x_train_new = np.vstack((x1_train[data1_idx, :], x2_train[data2_idx, :]))
    # y_train_new = np.hstack((y1_train[data1_idx], y2_train[data2_idx]))
    
else:
    x_train_new = np.vstack((x1_train, x2_train))
    y_train_new = np.hstack((y1_train, y2_train))


x_train_new = np.vstack((x1_train, x2_train))
y_train_new = np.hstack((y1_train, y2_train))


# Begin train the learning model
model_learn = GradientBoostingClassifier(random_state=0)
model_learn.fit(x_train_new, y_train_new)

accuracy1_total = []
accuracy2_total = []

data1_label = np.zeros(100)
data2_label = np.ones(100)


# model testing and adaptation
for i in range(100, data1.shape[0], 100):
    
    # print(i)
    
    x1_test = data1[i:i+100, :-1]
    y1_test = data1[i:i+100, -1]

    x2_test = data2[i:i+100, :-1]
    y2_test = data2[i:i+100, -1]
    
    
    # test the model on two data streams
    pred1_test = model_learn.predict(x1_test)
    pred2_test = model_learn.predict(x2_test)
    
    
    # calculate the accuracy of two results
    accuracy1_test = metrics.accuracy_score(y1_test, pred1_test)
    accuracy2_test = metrics.accuracy_score(y2_test, pred2_test)
    
    
    # print(accuracy1_test, accuracy2_test)
    accuracy1_total.append(accuracy1_test)
    accuracy2_total.append(accuracy2_test)
    
    
    # calculate the error
    eva1 = np.zeros(pred1_test.shape[0])
    right = 0
    for a in range(1, pred1_test.shape[0]):
        accuracy1 = metrics.accuracy_score(y1_test[0:a], pred1_test[0:a])
        eva1[a] = accuracy1
        
        # if pred1_test[a] == y1_test[a]:
        #     right = 1
        #     eva1[a] = right 
        # else:
        #     right = 0
        #     eva1[a] = right
        
    eva2 = np.zeros(pred2_test.shape[0])
    right = 0
    for b in range(1, pred2_test.shape[0]):
        accuracy2 = metrics.accuracy_score(y2_test[0:b], pred2_test[0:b])
        eva2[b] = accuracy2
        
        # if pred2_test[b] == y2_test[b]:
        #     right = 1
        #     eva2[b] = right 
        # else:
        #     right = 0
        #     eva2[b] = right
        
        
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
    data_count, index_del = segment(path)
    # print(index_del)
    # print(data_count)
    
    
    # drift detection
    drift_fre = 0
    data_count_drift = []
    for idx in range (1, len(data_count)):
        if s1[data_count[idx-1]] > s1[data_count[idx]]:
            drift_fre += 1
            data_count_drift.append(data_count[idx])
            
    print('Drift frequency:', drift_fre)
    # print('Data count drift:', data_count_drift)
    
    
    # identify drift region
    drift_list = region(path, data_count_drift)
    # print('Drift List:', drift_list)
    
    
    # data augmentation
    if len(index_del) > 0:
                    
        data2_idx = path_array[index_del, 1]
        data2_drift_idx = path_array[drift_list, 1]
        data2_drift_x = x2_test[data2_drift_idx, :]
        data2_drift_y = y2_test[data2_drift_idx]
        
        x_train_new = np.vstack((x1_test, x2_test[data2_idx, :]))
        y_train_new = np.hstack((y1_test, y2_test[data2_idx]))
        
        draw_region(x2_test, y2_test, data2_drift_x) 
        
    else:
        x_train_new = np.vstack((x1_test, x2_test))
        y_train_new = np.hstack((y1_test, y2_test))
        
    
    # x_train_new = np.vstack((x1_test, x2_test))
    # y_train_new = np.hstack((y1_test, y2_test))
    
    
    # retrain the learning model
    model_learn = GradientBoostingClassifier(random_state=0)
    model_learn.fit(x_train_new, y_train_new)



print('S1 accuracy:', np.average(accuracy1_total))
print('S2 accuracy:', np.average(accuracy2_total))





