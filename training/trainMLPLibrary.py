#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""


import cv2
import numpy as np
import fnmatch,os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import copy 
import pickle


            
model_type = 'top'

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
#object_list = ['010_potted_meat_can', '043_phillips_screwdriver', '038_padlock', '025_mug', '005_tomato_soup_can', '006_mustard_bottle', '024_bowl', '061_foam_brick_new', '037_scissors', '021_bleach_cleanser', '056_tennis_ball', '031_spoon', '019_pitcher_base', '052_extra_large_clamp', '029_plate', '048_hammer', '009_gelatin_box_new']
seq = 'allobjects'

if not os.path.exists('./libmodels/'+seq+'/'):
    os.mkdir('./libmodels/'+seq+'/')

for oname in object_list:
    alldata = np.load('./libpis/train1_library_allpis_'+oname+'.npy',allow_pickle=True).item()
    print(oname)
    maxlayers = 0
    instances = {}
    
    if oname == '035_power_drill':
        if model_type == 'top' or model_type=='all':
            bdeglist = list(set(cam_b_final) - set([10,15,20,170,165,160]))
        else:
            bdeglist = cam_b_final    
    else:
        bdeglist = cam_b_final

    for bdeg in  bdeglist:
        folder = str(bdeg)+'/0'
        for file in cam_a_final:
            for aug in range(4):
                instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = alldata[oname][0]['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)]
                
                
    data[oname] = (instances,alldata[oname][1])
    

#%%


## this is overall max layers... i.e max over all views front side and top 

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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

def classifier_mlp_softmax(n_classes=17,objlayers= 1):
    classifier = Sequential()
    classifier.add(Dense(512, input_shape = (1024*objlayers,)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(256))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))
    
    classifier.add(Dense(128))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(64))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.2))
    
    classifier.add(Dense(n_classes))
    classifier.add(BatchNormalization())
    classifier.add(Activation('softmax'))
    
    return classifier

def lr_schedule(epoch):
    lr = 1e-2
    if epoch > 100:
        lr *= 1e-1
    elif epoch > 50:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def getMLPInput(trainingData,numLayers):
    idxes = range(0,numLayers)
    datainput = trainingData[0]
    if len(idxes) == 1:
        return datainput
    else:
        for i in idxes[1:]:
            datainput = np.concatenate((datainput,trainingData[i]),axis=1)
        return datainput
        
from sklearn.model_selection import train_test_split



for l in range(1,overallMaxLayers+1): 
    xtrain, xtest, ytrain, ytest = train_test_split(np.nan_to_num(getMLPInput(trainingData,l)), labels, test_size=0.2, stratify=labels,random_state=2021)

    train_encoded_labels = to_categorical(ytrain)
    val_encoded_labels = to_categorical(ytest)
    
    model = classifier_mlp_softmax(len(object_list),l)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr_schedule(0)),metrics=['accuracy'])
    
    if not os.path.exists(os.path.dirname('./libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5')):
     	os.makedirs(os.path.dirname('./libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5'))
    if os.path.isfile('./libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5'):
        os.remove('./libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5')
    
    filepath = './libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5'

    checkpoint = ModelCheckpoint(filepath=filepath,
                                  monitor='val_accuracy',
                                  verbose=1,
                                  save_best_only=True)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                    cooldown=0,
                                    patience=10,
                                    min_lr=0.5e-6)
    
    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    
    
    history = model.fit(xtrain, train_encoded_labels,
     			batch_size=32,
    		epochs=100,
     			validation_data=(xtest, val_encoded_labels),
     			verbose=2,
     			shuffle=True,
                callbacks=callbacks)
    
    (loss, accuracy) = model.evaluate(xtest, val_encoded_labels,batch_size=64,verbose=1)
    print('[INFO] accuracy: {:.2f}%'.format(accuracy * 100))
    model.save_weights('./libmodels/'+seq+'/mlp_'+model_type+'_layer_'+str(l)+'.hdf5', overwrite=True)
