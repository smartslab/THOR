#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""


import cv2,argparse
import numpy as np
import fnmatch,os
import yaml
import json,shutil

def choosebetPredsTwoModelsConfidence(model_type_a,pred_a,conf_a, model_type_b,pred_b,conf_b,selectedareas,rc,printnames,object_list,objectRuleCodes):
        #is a possible
        #is b possible
        #if both possible go by confidence
        #if none possible go by confidence
    
    if pred_a == pred_b:
        predtoreturn = object_list[pred_a] if printnames else pred_a
    else:
        #start with both possible
        apossible = True
        bpossible=True
        #is a possible
        if rc not in objectRuleCodes[object_list[pred_a]]:
            apossible = False
        if rc not in objectRuleCodes[object_list[pred_b]]:
            bpossible = False
        
        
        if apossible and bpossible:
            if conf_a > conf_b:
                predtoreturn = object_list[pred_a] if printnames else pred_a
            else:
                predtoreturn = object_list[pred_b] if printnames else pred_b                 
        
        elif (apossible==False) and (bpossible==False):
            if conf_a > conf_b:
                predtoreturn = object_list[pred_a] if printnames else pred_a
            else:
                predtoreturn = object_list[pred_b] if printnames else pred_b              
        else:
            if apossible:
                predtoreturn = object_list[pred_a] if printnames else pred_a
            else:
                predtoreturn = object_list[pred_b] if printnames else pred_b
    return predtoreturn


def returnObjectRuleCodes():
    #these rules are obtained by using 20% threshold on actual object dimensions
    #i believe they should hold under my heavy occlusion assumption
    
      
    objectRuleCodes= {}
    objectRuleCodes['mustard_bottle'] = [4]
    objectRuleCodes['gelatin_box'] = [12,22,32]
    objectRuleCodes['bleach_cleanser'] = [4]
    objectRuleCodes['clamp'] = [4]
    objectRuleCodes['tennis_ball'] = [12,22,32]
    objectRuleCodes['tomato_soup'] = [11,21,31] 
    objectRuleCodes['foam_brick'] = [4]#[11,21,31]
    objectRuleCodes['pitcher_base'] = [11,21,31,4] #depends on whether handle is there or not
    objectRuleCodes['mug'] = [0,11,21,31] #depends on whether handle is there or not
    objectRuleCodes['screw_driver'] = [11,21,31]
    objectRuleCodes['hammer'] = [4]
    objectRuleCodes['scissors'] = [14]
    objectRuleCodes['padlock'] = [4]
    objectRuleCodes['plate'] = [12,22,32]
    objectRuleCodes['bowl'] = [12,22,32]
    objectRuleCodes['potted_meat_can'] = [12,22,32]
    objectRuleCodes['spoon'] = [4]
    
    return objectRuleCodes



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
    


def maintest3(videodir,environment,category,separation,light):
    if not os.path.exists('./predictions/'):
        os.mkdir('./predictions/')    
    if not os.path.exists('./groundtruth/'):
        os.mkdir('./groundtruth/') 
    if not os.path.exists('./predictions/mlp/'):
        os.mkdir('./predictions/mlp/')
    intvideolist = returnvideolist(environment,category,separation,light)

    videolist = [str(x) for x in intvideolist]

    object_list = ['potted_meat_can', 'screw_driver', 'padlock', 'mug', 'tomato_soup', 'mustard_bottle', 'bowl', 'foam_brick', 'scissors', 'bleach_cleanser', 'tennis_ball', 'spoon', 'pitcher_base', 'clamp', 'plate', 'hammer', 'gelatin_box']

    printnames = False
    for video in videolist:
        print(video)
        allpreds = []
        true = []

        instances = sorted(fnmatch.filter(os.listdir(videodir +video+'/images/'),'*.yaml'))
        errors = fnmatch.filter(os.listdir('./temp/deptherrorlog/'+video+'/'),'*.txt')
        
        for i in instances:
            fileid = i.split('_')[0]
            label = cv2.imread(videodir +video+'/images/'+fileid+'_labels.png',0)
            groundtruthdict = getGroundTruthFromYAML(videodir +video+'/images/'+fileid+'_poses.yaml')
            labelidxes = [i for i in np.unique(label) if i > 0]
            
            for idx in labelidxes:
                if groundtruthdict[idx] not in ['ice_cream','hot_sauce','chips_can']:
                    if fileid+'_idx'+str(idx)+'#'+groundtruthdict[idx]+'.txt' not in errors:

                        with open('./temp/savedinjson/'+video+'/'+fileid+'_idx'+str(idx)+'.json','r') as read:
                            dump = json.load(read)

                        l = dump['l']
                        modeltypes = dump['modeltypes']
                        selectedareas = dump['selectedareas']
                        returncode = dump['returncode']


                        predictions = []
                        predprobs = []

                        for mtype in modeltypes:
                            result = np.load('./temp/mlpmidresults/'+mtype+'/'+str(l+1)+'layers/'+video+'/'+fileid+'_idx'+str(idx)+'.npy')
                            predictions.append(int(result[0,0]))
                            predprobs.append(result[0,1])
                            
                        if len(modeltypes) > 1:
                            objectRuleCodes = returnObjectRuleCodes()
                            pred = choosebetPredsTwoModelsConfidence(modeltypes[0],predictions[0],predprobs[0], modeltypes[1],predictions[1],predprobs[1],selectedareas,returncode,printnames,object_list,objectRuleCodes)
                            allpreds.append(pred)
                        else:
                            if printnames:
                                pred = object_list[predictions[0]]
                                allpreds.append(pred)
                            else:
                                pred = predictions[0]
                                allpreds.append(pred)
                        true.append(object_list.index(groundtruthdict[idx]))

                
        with open('./predictions/mlp/'+video+'.txt', "w") as output:
            output.write(str(allpreds))    
        with open('./groundtruth/'+video+'.txt', "w") as toutput:
            toutput.write(str(true))         
        shutil.rmtree('./temp/savedinjson/'+video+'/')
        for deletemodeltype in ['front','side','top']:
            for layernumdelete in range(0,7):
                shutil.rmtree('./temp/mlpmidresults/'+deletemodeltype+'/'+str(layernumdelete+1)+'layers/'+video+'/')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir')
    parser.add_argument('--environment')
    parser.add_argument('--category')
    parser.add_argument('--separation')
    parser.add_argument('--light')   
    args = parser.parse_args()
    maintest3(args.videodir,args.environment,args.category,args.separation,args.light)