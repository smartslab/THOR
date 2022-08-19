#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""

import cv2
import numpy as np
import open3d as o3d
import fnmatch,os
import matplotlib.pyplot as plt
import copy 
from gtda.plotting import plot_point_cloud
import pickle
from scipy.spatial.transform import Rotation

from persim import PersistenceImager


            
model_type = 'side'

print(model_type)
if model_type == 'front':
    cam_a = [i for i in range(0,55,5)]+[i for i in range(130,235,5)] + [i for i in range(310,360,5)]
    cam_b = [i for i in range(40,145,5)]
    cam_a_remove = []
    cam_b_remove = []
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
if model_type == 'side':
    cam_a = [i for i in range(40,145,5)]+[i for i in range(220,325,5)]
    cam_b = [i for i in range(40,145,5)]
    cam_a_remove = []
    cam_b_remove = []
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
if model_type == 'top':
    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,55,5)] + [i for i in range(130,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]    
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
    
if model_type == 'all':
    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]     
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
            
            

data = {}

object_list = os.listdir('./library/')


seq = 'allobjects'

# if not os.path.exists('./libmodels/'+seq+'/'):
#     os.mkdir('./libmodels/'+seq+'/')
for oname in object_list:
    alldata = np.load('./libpis/train1_library_allpis_'+oname+'.npy',allow_pickle=True).item()
    print(oname)
    maxlayers = 0
    instances = {}


    for bdeg in  cam_b_final:
        folder = str(bdeg)+'/0'
        for file in cam_a_final:
            for aug in range(4):
                instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = alldata[oname][0]['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)]
                
                
    data[oname] = (instances,alldata[oname][1])
    

#%%


## this is overall max layers...i.e. max over all views front side and top 

overallMaxLayers = 0
for key,value in data.items():
    overallMaxLayers = max(value[1],overallMaxLayers)
   
#%%

def getLthPIs(data,l):
    pi = []
    for key,value in data.items():
        for k,v in value[0].items():
            if l in v:
                pi.append(np.reshape(value[0][k][l],(1024,)))
            else:
                pi.append(np.ones((1024,)))           
    return pi

def getLabels(data):
    labels = []
    for key,value in data.items():
        for k,v in value[0].items():
            labels.append(object_list.index(key))
    return labels
#%%
trainingData = {}

for l in range(overallMaxLayers):
    trainingData[l] = np.asarray(getLthPIs(data,l))

labels = getLabels(data)
#%%    

def getSVMInput(trainingData,numLayers):
    idxes = range(0,numLayers)
    datainput = trainingData[0]
    if len(idxes) == 1:
        return datainput
    else:
        for i in idxes[1:]:
            datainput = np.concatenate((datainput,trainingData[i]),axis=1)
        return datainput
        

from sklearn.model_selection import train_test_split



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

for l in range(1,overallMaxLayers+1): 
    #xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2021)
    #xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2022) 
    #xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2023)
    xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2019)
    #xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2020)
    base_clf = SVC(gamma='auto')
            

    clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(base_estimator=base_clf, cv=3))  
    clf.fit(xtrain, ytrain)
    print('SVM',round(clf.score(xtest,ytest),4))   
    filename = './libmodels2019/'+seq+'/'+model_type+'_layer_'+str(l)+'.sav'
    pickle.dump(clf,open(filename,'wb'))
