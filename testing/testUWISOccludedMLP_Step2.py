#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""

import cv2
import numpy as np
import fnmatch,os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import argparse,shutil

def classifier_mlp_softmax(n_classes=8,objlayers= 1):
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


def loadMLP(modeldir,layers,model_type):
    model = classifier_mlp_softmax(17,layers)
    model.load_weights(modeldir+model_type+'_layer_'+str(layers)+'.hdf5')
    return model
                
                            


def getGroundTruthFromYAML(file):
    with open(file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            labels = {}
            for key,value in data.items():
                labels[value['label']] = key
        except yaml.YAMLError as exc:
            print(exc)
    return labels            

def returnvideolist(environment,category,separation,light):

    o1kitchenl1 = [41,42,176,177]
    o1foodl1 = [32,33,167,168]
    o1toolsl1 =[50,51,185,186]
    o2kitchenl1 = [44,45,179,180]
    o2foodl1 = [35,36,170,171]
    o2toolsl1 = [53,54,188,189]
    o3kitchenl1 = [46,48,182,183]
    o3foodl1 = [38,39,173,174]
    o3toolsl1 = [56,57,191,192]

    o1kitchenl2 = [68,69,203,204]
    o1foodl2 = [77,78,212,213]
    o1toolsl2 =[59,60,194,195]
    o2kitchenl2 = [71,72,206,207]
    o2foodl2 = [80,81,215,216]
    o2toolsl2 = [62,63,197,198]
    o3kitchenl2 = [74,75,209,210]
    o3foodl2 = [83,84,218,219]
    o3toolsl2 = [65,66,200,201]


    if light == '1':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl1
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl1
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl1
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl1 + o2kitchenl1 + o3kitchenl1
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl1
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl1
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl1
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl1 + o2foodl1 + o3foodl1
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl1
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl1
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl1
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl1 + o2toolsl1 + o3toolsl1
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl1 + o1foodl1 + o1kitchenl1
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl1 + o2foodl1 + o2kitchenl1
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl1 + o3foodl1 + o3kitchenl1
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl1 + o1foodl1+o1toolsl1 +o2kitchenl1 + o2foodl1+o2toolsl1+o3kitchenl1 + o3foodl1+o3toolsl1
        else:
            raise NotImplementedError
    elif light == '2':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl2
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl2
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl2
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o2kitchenl2 + o3kitchenl2
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl2
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl2
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl2
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl2 + o2foodl2 + o3foodl2
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl2
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl2
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl2
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl2 + o2toolsl2 + o3toolsl2
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl2 + o1foodl2 + o1kitchenl2
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl2 + o2foodl2 + o2kitchenl2
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl2 + o3foodl2 + o3kitchenl2
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o1foodl2+o1toolsl2 +o2kitchenl2 + o2foodl2+o2toolsl2+o3kitchenl2 + o3foodl2+o3toolsl2
        else:
            raise NotImplementedError
    elif light == 'both':
        if category == 'kitchen' and separation == 'level1':
            intvideolist = o1kitchenl1+o1kitchenl2
        elif category == 'kitchen' and separation == 'level2':
            intvideolist = o2kitchenl1+o2kitchenl2
        elif category == 'kitchen' and separation == 'level3':
            intvideolist = o3kitchenl1+o3kitchenl2
        elif category == 'kitchen' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o2kitchenl2 + o3kitchenl2 + o1kitchenl1 + o2kitchenl1 + o3kitchenl1
        
        elif category == 'food' and separation == 'level1':
            intvideolist = o1foodl1+o1foodl2
        elif category == 'food' and separation == 'level2':
            intvideolist = o2foodl2+o2foodl1
        elif category == 'food' and separation == 'level3':
            intvideolist = o3foodl2+o3foodl1
        elif category == 'food' and separation == 'alllevels':
            intvideolist = o1foodl2 + o2foodl2 + o3foodl2+o1foodl1 + o2foodl1 + o3foodl1
            
        elif category == 'tools' and separation == 'level1':
            intvideolist = o1toolsl2+o1toolsl1
        elif category == 'tools' and separation == 'level2':
            intvideolist = o2toolsl2+o2toolsl1
        elif category == 'tools' and separation == 'level3':
            intvideolist = o3toolsl2+o3toolsl1
        elif category == 'tools' and separation == 'alllevels':
            intvideolist = o1toolsl2 + o2toolsl2 + o3toolsl2+o1toolsl1 + o2toolsl1 + o3toolsl1
            
        elif category == 'all' and separation == 'level1':
            intvideolist = o1toolsl2 + o1foodl2 + o1kitchenl2 + o1toolsl1 + o1foodl1 + o1kitchenl1
        elif category == 'all' and separation == 'level2':
            intvideolist = o2toolsl2 + o2foodl2 + o2kitchenl2 + o2toolsl1 + o2foodl1 + o2kitchenl1
        elif category == 'all' and separation == 'level3':
            intvideolist = o3toolsl2 + o3foodl2 + o3kitchenl2 + o3toolsl1 + o3foodl1 + o3kitchenl1
        elif category == 'all' and separation == 'alllevels':
            intvideolist = o1kitchenl2 + o1foodl2+o1toolsl2 +o2kitchenl2 + o2foodl2+o2toolsl2+o3kitchenl2 + o3foodl2+o3toolsl2 + o1kitchenl1 + o1foodl1+o1toolsl1 +o2kitchenl1 + o2foodl1+o2toolsl1+o3kitchenl1 + o3foodl1+o3toolsl1
            
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
        
    if environment == 'warehouse':    
        intvideolist = [x for x in intvideolist if x < 167]
    elif environment == 'lounge':
        intvideolist = [x for x in intvideolist if x > 166]
    elif environment == 'both':
        intvideolist = intvideolist
    else:
        raise NotImplementedError

    return intvideolist

def mainstep2(videodir, environment, category, separation, light, model_dir, lstr, model_set):

    l = int(lstr)
    model_type = model_set
    if not os.path.exists('./temp/mlpmidresults/'):
        os.mkdir('./temp/mlpmidresults/') 
    if not os.path.exists('./temp/mlpmidresults/'+model_type):
        os.mkdir('./temp/mlpmidresults/'+model_type) 
    if not os.path.exists('./temp/mlpmidresults/'+model_type+'/'+str(l+1)+'layers/'):
        os.mkdir('./temp/mlpmidresults/'+model_type+'/'+str(l+1)+'layers/') 
        
    intvideolist = returnvideolist(environment,category,separation,light)
    
    
    videolist = [str(x) for x in intvideolist]

    object_list = ['potted_meat_can', 'screw_driver', 'padlock', 'mug', 'tomato_soup', 'mustard_bottle', 'bowl', 'foam_brick', 'scissors', 'bleach_cleanser', 'tennis_ball', 'spoon', 'pitcher_base', 'clamp', 'plate', 'hammer', 'gelatin_box']
         
          
    model = loadMLP(model_dir,l+1,model_type)
                                
      
    for video in videolist:
        if not os.path.exists('./temp/mlpmidresults/'+model_type+'/'+str(l+1)+'layers/'+video):
            os.mkdir('./temp/mlpmidresults/'+model_type+'/'+str(l+1)+'layers/'+video)         
        instances = sorted(fnmatch.filter(os.listdir(videodir +video+'/images/'),'*.yaml'))
        errors = fnmatch.filter(os.listdir('./temp/deptherrorlog/'+video+'/'),'*.txt')
        numobjinstances = len(fnmatch.filter(os.listdir('./temp/topsfeatures/'+video+'/'),'*.npy'))
        featuresarray = np.zeros((numobjinstances,1024*(l+1)))
        ctr = 0
        for i in instances:
            #print(i, ' in ', video)
            #print(np.shape(np.asarray(pcds)))
            fileid = i.split('_')[0]
            label = cv2.imread(videodir +video+'/images/'+fileid+'_labels.png',0)
            groundtruthdict = getGroundTruthFromYAML(videodir +video+'/images/'+fileid+'_poses.yaml')
            labelidxes = [k for k in np.unique(label) if k > 0]

            for idx in labelidxes:
                if groundtruthdict[idx] not in ['ice_cream','hot_sauce','chips_can']:
                    if fileid+'_idx'+str(idx)+'#'+groundtruthdict[idx]+'.txt' not in errors:
                        feature = np.load('./temp/topsfeatures/'+video+'/'+ fileid+'_idx'+str(idx)+'.npy')

                        inputfeature = np.nan_to_num(feature[:,:1024*(l+1)])
                        featuresarray[ctr,:] = inputfeature
                        ctr = ctr+1 
                        #print(ctr)
        
        probsarray = model.predict(featuresarray)

        ctrctr = 0
        for ii in instances:
            #print(ii, ' in ', video)
            #print(np.shape(np.asarray(pcds)))
            newfileid = ii.split('_')[0]
            label = cv2.imread(videodir +video+'/images/'+newfileid+'_labels.png',0)
            groundtruthdict = getGroundTruthFromYAML(videodir +video+'/images/'+newfileid+'_poses.yaml')
            labelidxes = [kk for kk in np.unique(label) if kk > 0]

            allpreds = {}
            allprobs = {}
            allactclassprobs = {}
            
            
            for idx in labelidxes:
                if groundtruthdict[idx] not in ['ice_cream','hot_sauce','chips_can']:
                    if newfileid+'_idx'+str(idx)+'#'+groundtruthdict[idx]+'.txt' not in errors:
                        
                        predictions = []
                        predprobs = []
                        actclassprobs = []

                        probs = probsarray[ctrctr:ctrctr+1,:]

                        predictions.append(np.argmax(probs))
                        predprobs.append(np.max(probs))
                        actclassprobs.append(np.asarray(probs)[:,object_list.index(groundtruthdict[idx])][0])

                        allpreds[groundtruthdict[idx]] = np.expand_dims(np.asarray(predictions),axis=1)
                        allprobs[groundtruthdict[idx]] = np.expand_dims(np.asarray(predprobs),axis=1)
                        allactclassprobs[groundtruthdict[idx]] = np.expand_dims(np.asarray(actclassprobs),axis=1)
                        storethis = np.concatenate((allpreds[groundtruthdict[idx]],allprobs[groundtruthdict[idx]],allactclassprobs[groundtruthdict[idx]]),axis=1)
                        
                        np.save('./temp/mlpmidresults/'+model_type+'/'+str(l+1)+'layers/'+video+'/'+newfileid+'_idx'+str(idx)+'.npy',storethis)
                        ctrctr = ctrctr+1
        shutil.rmtree('./temp/topsfeatures/'+video+'/')



                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir')
    parser.add_argument('--environment')
    parser.add_argument('--category')
    parser.add_argument('--separation')
    parser.add_argument('--light')
    parser.add_argument('--models_dir')  
    parser.add_argument('--layer')
    parser.add_argument('--model_set')
    args = parser.parse_args()
    mainstep2(args.videodir, args.environment, args.category, args.separation, args.light,args.model_dir, args.layer, args.model_set)
