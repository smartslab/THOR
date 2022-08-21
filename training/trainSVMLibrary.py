#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""

import numpy as np
import os,argparse
import pickle

from sklearn.model_selection import train_test_split



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

def getLthPIs(data,l):
    pi = []
    for key,value in data.items():
        for k,v in value[0].items():
            if l in v:
                pi.append(np.reshape(value[0][k][l],(1024,)))
            else:
                pi.append(np.ones((1024,)))           
    return pi

def getLabels(data,object_list):
    labels = []
    for key,value in data.items():
        for k,v in value[0].items():
            labels.append(object_list.index(key))
    return labels

def getSVMInput(trainingData,numLayers):
    idxes = range(0,numLayers)
    datainput = trainingData[0]
    if len(idxes) == 1:
        return datainput
    else:
        for i in idxes[1:]:
            datainput = np.concatenate((datainput,trainingData[i]),axis=1)
        return datainput

def main(data_path,pis_path,model_set,random_state):
            
    model_type = model_set
    random_state = int(random_state)
    print('Training models for model set: ', model_type)
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
        
            

    data = {}
    
    object_list = os.listdir(data_path)


    if not os.path.exists('./librarymodels/'):
        os.mkdir('./librarymodels/')    
    if not os.path.exists('./librarymodels/svm/'):
        os.mkdir('./librarymodels/svm/')
    if not os.path.exists('./librarymodels/svm/'+random_state+'/'):
        os.mkdir('./librarymodels/svm/'+random_state+'/')
    for oname in object_list:
        alldata = np.load('./libpis/allpis_'+oname+'.npy',allow_pickle=True).item()
        instances = {}

        for bdeg in  cam_b_final:
            for file in cam_a_final:
                for aug in range(4):
                    instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = alldata[oname][0]['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)]
        
        data[oname] = (instances,alldata[oname][1])
    


    overallMaxLayers = 0
    for key,value in data.items():
        overallMaxLayers = max(value[1],overallMaxLayers)
   

    trainingData = {}
    
    for l in range(overallMaxLayers):
        trainingData[l] = np.asarray(getLthPIs(data,l))
    
    labels = getLabels(data,object_list)


        



    for l in range(1,overallMaxLayers+1): 
        xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getSVMInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=random_state)
        base_clf = SVC(gamma='auto')
        clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(base_estimator=base_clf, cv=3))  
        clf.fit(xtrain, ytrain)
        print('SVM for layer ',l, 'with accuracy: ', round(clf.score(xtest,ytest),4))   
        filename = './librarymodels/svm/'+random_state+'/'+model_type+'_layer_'+str(l)+'.sav'
        pickle.dump(clf,open(filename,'wb'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--pis_path')
    parser.add_argument('--model_set')
    parser.add_argument('--random_state')
    
    args = parser.parse_args()
    main(args.data_path,args.pis_path,args.model_set,args.random_state)