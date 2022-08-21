#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""

import argparse
import numpy as np
import open3d as o3d
import os
import copy 
from persim import PersistenceImager


pimgr = PersistenceImager()
pimgr.birth_range = (0,0.75)
pimgr.pers_range = (0,0.75)
pimgr.kernel_params = {'sigma': 0.00025}
pimgr.pixel_size = 0.025


def removeNANs(pcd):
    a = np.asarray(pcd.points)
    pts = a[~np.isnan(a).any(axis=1)]
    newpcd = o3d.PointCloud()
    newpcd.points = o3d.utility.Vector3dVector(pts)
    return newpcd

def roundinXYZ(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = np.round(pcdpts[:,0],2)
    pcdpts[:,1] = np.round(pcdpts[:,1],2)
    pcdpts[:,2] = np.round(pcdpts[:,2],1)
    return pcdpts

def getZs(pcdpts):
    zlist = sorted(np.unique(pcdpts[:,2]))
    zs = {}
    for idx,num in enumerate(zlist):
        zs[idx] = num
    return zs

def getLayer(pcdpts,zs,layeridx):
    return pcdpts[np.where(pcdpts[:,2] == zs[layeridx])]


def computePDBinningNo2DTranslation(pcd):
    #binning
    bins = np.arange(0,0.775,0.025)
    xes = copy.deepcopy(pcd[:,0])
    pcd[:,0] = bins[np.digitize(xes,bins,right=True)]
    xesnew = np.unique(pcd[:,0])
    dgm = []
    for idx,x in enumerate(xesnew):
        ymax = np.max(pcd[np.where(pcd[:,0] == x)][:,1])
        ymin = np.min(pcd[np.where(pcd[:,0] == x)][:,1])

        dgm.append((x,x+ymax-ymin,0))

    return np.asarray(dgm)



def flipX(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = -pcdpts[:,0]
    fac = np.min(pcdpts[:,0])

    pcdpts[:,0] = -fac + pcdpts[:,0]
    return pcdpts

def flipY(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,1] = -pcdpts[:,1]
    fac = np.min(pcdpts[:,1])

    pcdpts[:,1] = -fac + pcdpts[:,1]
    return pcdpts


def trZMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([0,0,-np.min(pts[:,2])])
    return pcd

def trYMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([0,-np.min(pts[:,1]),0])
    return pcd

def trXMinusCam(pcd):
    pts = np.asarray(pcd.points)[:-2,:]
    pcd.translate([-np.min(pts[:,0]),0,0])
    return pcd
 

def rotateForLayeringOption2WAug(pcd,aug):

    pcdpts = np.asarray(pcd.points)[:-2,:]

    if aug == 0:
        pts = pcdpts
    elif aug == 1:
        pts = flipX(pcdpts)
    elif aug == 2:
        pts = flipY(pcdpts)
    else:
        pts = flipX(flipY(pcdpts))
    
    newpcd = o3d.geometry.PointCloud()
    newpcd.points = o3d.utility.Vector3dVector(pts)
    
    R45 = o3d.geometry.get_rotation_matrix_from_xyz([0,-np.pi/4,0])
    newpcd.rotate(R45)

    return newpcd 

def orientCamBottom(pcd):
    campos = np.asarray(pcd.points)[-1,:] 
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    return pcd
            

def main(data_path):
    pidir = './training/libpis/'
    os.mkdir(pidir)
    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]     
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))

    object_list = os.listdir(data_path)
    for oname in object_list:
        data = {} 
        print(oname)
        maxlayers = 0
        instances = {}
    
        for bdeg in  cam_b_final:
            folder = str(bdeg)+'/0'
            for file in cam_a_final:
                for aug in range(4):
                    pcd = o3d.io.read_point_cloud(data_path+oname+'/'+folder+'/'+'vnpcdwcam/'+str(file)+'.pcd')
                    
                    trpcd = trXMinusCam(trYMinusCam(trZMinusCam(pcd)))
                    rotatedpcd = orientCamBottom(trpcd)
                    finaltrpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))
    
                    rotatedpcd = rotateForLayeringOption2WAug(finaltrpcd,aug)
                    finalpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))
    
                    pcdpts = np.asarray(finalpcd.points)
    
    
            
                    rounded = roundinXYZ(pcdpts)
    
                    
                    zs = getZs(rounded)
                    pis = {}
                    for key,value in zs.items():
                        layer = getLayer(rounded,zs,key)
                        dgm = computePDBinningNo2DTranslation(layer)
    
                        pers = dgm[:,1] - dgm[:,0]
                        if (pers > 0.75).any():
                            print('pers range issue')
                        img = pimgr.transform([dgm[:,0:2]])
                        pis[key] = img[0]
                    maxlayers = max(maxlayers,key+1)
                    instances['a'+str(aug)+'_'+str(bdeg)+'_'+str(file)] = pis
        data[oname] = (instances,maxlayers)
        np.save(pidir+'allpis_'+oname+'.npy',data)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    main(args.data_path)
