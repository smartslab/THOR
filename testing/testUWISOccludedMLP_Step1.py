#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""

import cv2,argparse
import numpy as np
import open3d as o3d
import fnmatch,os
import copy 
from persim import PersistenceImager
from scipy.stats import iqr
import yaml
import shutil,json


pimgr = PersistenceImager()
pimgr.birth_range = (0,0.75)
pimgr.pers_range = (0,0.75)
pimgr.kernel_params = {'sigma': 0.00025}
pimgr.pixel_size = 0.025


def scaleObjectPCD(pcd,scaleFactor):
    scaled = copy.deepcopy(pcd)
    scaled.scale(scaleFactor,center=[0,0,0])
    return scaled


def getObjectPCD(pcd,objidx,label,color):
    label1d = np.reshape(label,(np.shape(label)[0]*np.shape(label)[1],))
    color1d = np.reshape(color,(np.shape(label)[0]*np.shape(label)[1],))/255
    idxes = np.where(label1d==objidx)
    allpts = np.asarray(pcd.points)
    objpts = allpts[idxes]
    colorpts = np.expand_dims(color1d[idxes],axis=1)
    objptscolor = np.concatenate((colorpts,np.zeros_like(colorpts),np.zeros_like(colorpts)),axis=1)
    objpcd = o3d.geometry.PointCloud()
    objpcd.points = o3d.utility.Vector3dVector(objpts)
    objpcd.colors = o3d.utility.Vector3dVector(objptscolor)
    return objpcd

def getObjectOcclusionColors(badidxes, objidx,label,color):
    editedlabel=copy.deepcopy(label)
    editedlabel[badidxes] = -1 
    label1d = np.reshape(editedlabel,(np.shape(label)[0]*np.shape(label)[1],))
    color1d = np.reshape(color,(np.shape(label)[0]*np.shape(label)[1],))/255
    idxes = np.where(label1d==objidx)
    colorpts = np.expand_dims(color1d[idxes],axis=1)
    objptscolor = np.concatenate((colorpts,np.zeros_like(colorpts),np.zeros_like(colorpts)),axis=1)
    return objptscolor
    
def findContour(objidx,label):
    binimg = np.expand_dims(255*np.where(label==objidx,1,0),axis=2).astype('uint8')
    contours,_ = cv2.findContours(binimg,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea)
    return np.squeeze(cnt[-1])[:,::-1] #because opencv flips rows and columns



def checkOccludeeContour(contour,objidx,label,depth):
    boundary = np.zeros_like(label)
    for i in range(np.shape(contour)[0]):
        x,y = contour[i,:]
        if x == 719:
            boundary[x,y] = 255 ## if object is cut
        elif y == 1279:
            boundary[x,y] = 255
        else:
            if label[x-1,y] > 0 and label[x-1,y] != objidx:
                if depth[x-1,y] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x+1,y] > 0 and label[x+1,y] != objidx:
                if depth[x+1,y] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x,y-1] > 0 and label[x,y-1] != objidx:
                if depth[x,y-1] < depth[x,y]:
                    boundary[x,y] = 255
            if label[x,y+1] > 0 and label[x,y+1] != objidx:
                if depth[x,y+1] < depth[x,y]:
                    boundary[x,y] = 255
    return boundary

def rotateToFlatForLayering(pcd):
    pcdpts = np.asarray(pcd.points)[:-2,:]
    bbox = o3d.geometry.OrientedBoundingBox()
    bboxresult = bbox.create_from_points(o3d.utility.Vector3dVector(pcdpts))#,robust=True)
    Rnew = np.transpose(bboxresult.R)
    pcd.rotate(Rnew)
    

    #the angle is wrt to point with highest y. cw rotation of x axis until it meets an edge of bb
    w2,h2,angle2 = get2dboundingboxXYEfficient(np.asarray(pcd.points)[:-2,:])
    
    if h2 < w2:
        angles = [0,0, (angle2*np.pi)/180]
        R2dxy = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dxy)
    else:
        angle2 = 90-angle2
        angles = [0,0,-(angle2*np.pi)/180]
        R2dxy = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dxy)    


    w1,h1,angle1 = get2dboundingboxYZEfficient(np.asarray(pcd.points)[:-2,:])
    
    if h1 < w1:
        angles = [(angle1*np.pi)/180,0,0]
        R2dyz = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dyz)
    else:
        angle1 = 90-angle1
        angles = [-(angle1*np.pi)/180,0,0]
        R2dyz = o3d.geometry.get_rotation_matrix_from_xyz(angles)
        pcd.rotate(R2dyz)
        
    campos = np.asarray(pcd.points)[-1,:]
    
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    
    pts = np.asarray(pcd.points)[:-2,:]
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bboxresult = bbox.create_from_points(o3d.utility.Vector3dVector(pts))#,robust=True)
    extent = bboxresult.get_extent()  
    
    return pcd,extent

def get2dboundingboxXYEfficient(points):
    final = my_scatter_plot_xy(points)
    final = final.astype(np.uint8) 
    kernel = np.ones((11, 11), np.uint8)
    final = cv2.erode(final, kernel, cv2.BORDER_REFLECT) 

    
    _,thresh = cv2.threshold(final,127,255,cv2.THRESH_BINARY_INV)

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt[-1]) 
    (x,y),(w,h), a = rect # a - angle
    
    return w,h,a

def get2dboundingboxYZEfficient(points):
    final = my_scatter_plot_yz(points)
    final = final.astype(np.uint8) 
    kernel = np.ones((11, 11), np.uint8)

    final = cv2.erode(final, kernel, cv2.BORDER_REFLECT) 
    
    
    _,thresh = cv2.threshold(final,127,255,cv2.THRESH_BINARY_INV)
    

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt[-1]) 

    (x,y),(w,h), a = rect # a - angle

    return w,h,a

def my_scatter_plot_xy(points):
    nx,ny = (224,224)
    xmin = np.min(points[:,0])
    ymin = np.min(points[:,1])
    img = 255*np.ones((nx,ny))
    
    x = np.linspace(xmin - 0.1,xmin+0.9,nx)
    y = np.linspace(ymin-0.1,ymin+0.9,ny)
    xbins = np.digitize(points[:,0],x)
    ybins = len(y) - np.digitize(points[:,1],y)
    
    for i in range(len(points)):
        img[ybins[i],xbins[i]] = 0
    
    return img

def my_scatter_plot_yz(points):
    ny,nz = (224,224)
    ymin = np.min(points[:,1])
    zmin = np.min(points[:,2])
    img = 255*np.ones((ny,nz))
    
    y = np.linspace(ymin - 0.1,ymin+0.9,ny)
    z = np.linspace(zmin-0.1,zmin+0.9,nz)
    ybins = np.digitize(points[:,1],y)
    zbins = len(z) - np.digitize(points[:,2],z)
    
    for i in range(len(points)):
        img[zbins[i],ybins[i]] = 0
    
    return img   



def clusteringNoCurvature(newnormals,points,refnormals):

#do nothing with colors because they store occlusion info
              
    labels = []
    facecounts = []
    faces = {}
    for i in range(len(newnormals)):
        angles = np.arccos(np.clip(np.dot(refnormals,newnormals[i,:]),-1.0,1.0))
        lid = np.argmin(angles)
        labels.append(lid)
    
    labels = np.asarray(labels)
    faceids,counts = np.unique(labels,return_counts=True)
    
    for f in range(6):
        if f not in faceids:
            facecounts.append(0)
            faces[f] = np.array([])
        else:
            idxes = np.where(labels==f)
            faces[f] = points[idxes] 
            #can't use dictionaries to avoid cases of same #points in two faces etc
            facecounts.append(np.shape(faces[f])[0]) 
        
    return labels,faces,facecounts


def getGrad(points,values,f):
    values = np.asarray(values)
    grads1 = np.zeros_like(values)
    grads2 = np.zeros_like(values)
    rounded = np.round(copy.deepcopy(points),2)
    xs = np.unique(rounded[:,0])
    ys = np.unique(rounded[:,1])
    zs = np.unique(rounded[:,2])
    
    if f==0 or f==3:

        for z in zs:
            idxes = np.where(rounded[:,2]==z)
            if np.shape(idxes[0])[0] > 1:
                
                inpy = np.squeeze(points[idxes,1],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpy)
                coeffn = np.diff(inpy)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,0],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads1[idxes] = pd*(coeffn/coeffd)
        for y in ys:
            idxes = np.where(rounded[:,1]==y)
            if np.shape(idxes[0])[0] > 1:
                inpz = np.squeeze(points[idxes,2],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpz)
                coeffn = np.diff(inpz)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,0],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads2[idxes] = pd*(coeffn/coeffd)
            
    elif f==1 or f==4:

        for z in zs:
            idxes = np.where(rounded[:,2]==z)
            if np.shape(idxes[0])[0] > 1:
                inpx = np.squeeze(points[idxes,0],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpx)
                coeffn = np.diff(inpx)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,1],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads1[idxes] = pd*(coeffn/coeffd)
        for x in xs:
            idxes = np.where(rounded[:,0]==x)
            if np.shape(idxes[0])[0] > 1:
                inpz = np.squeeze(points[idxes,2],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpz)
                coeffn = np.diff(inpz)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,1],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads2[idxes] = pd*(coeffn/coeffd)
    else:

        for y in ys:
            idxes = np.where(rounded[:,1]==y)
            if np.shape(idxes[0])[0] > 1:
                inpx = np.squeeze(points[idxes,0],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpx)
                coeffn = np.diff(inpx)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,2],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads1[idxes] = pd*(coeffn/coeffd)
        for x in xs:
            idxes = np.where(rounded[:,0]==x)
            if np.shape(idxes[0])[0] > 1:
                inpy = np.squeeze(points[idxes,1],axis=0)
                vals = values[idxes]
                pd = np.gradient(vals,inpy)
                coeffn = np.diff(inpy)
                coeffn = np.concatenate((coeffn,np.array([coeffn[-1]])),axis=0)
                coeffd = np.diff(np.squeeze(points[idxes,2],axis=0))
                coeffd = np.concatenate((coeffd,np.array([coeffd[-1]])),axis=0)
                grads2[idxes] = pd*(coeffn/coeffd)
            
    return (grads1+grads2)
            
def getGradRep(grads):
    grads = abs(grads)
    gradrep = (np.mean(grads),np.std(grads),np.nan_to_num(np.std(grads)/np.mean(grads)),iqr(grads))
    return gradrep

def getCurvatureKDTreeFaceWGrad(points,f,radius=0.05):

    from scipy.stats import iqr
    from scipy.spatial import KDTree
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]

    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)
        
        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        #M = np.cov(M)
        M = np.nan_to_num(np.cov(M))

        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V
        if (h1+h2+h3) !=0:
            curvature[index] = h3 / (h1 + h2 + h3)
        #curvature[index] = h3 / (h1 + h2 + h3)
    grads = getGrad(points,curvature,f)
    curverep= (np.mean(curvature),np.std(curvature),np.nan_to_num(np.std(curvature)/np.mean(curvature)),iqr(curvature))
    return curverep,grads

def roundinXYZ(pts):
    pcdpts = copy.deepcopy(pts)
    pcdpts[:,0] = np.round(pcdpts[:,0],2)
    pcdpts[:,1] = np.round(pcdpts[:,1],2)
    pcdpts[:,2] = np.round(pcdpts[:,2],1)
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

def checkNeedToFlipMinusCam(flatpcd):
    flatpts = np.asarray(flatpcd.points)[:-2,:]
    xmin = np.min(flatpts[:,0])
    ymin = np.min(flatpts[:,1])
    zmin = np.min(flatpts[:,2])
    points = flatpts - [xmin,ymin,zmin]
    
    xmax = np.max(points[:,0])                
    threecolors = np.asarray(flatpcd.colors)[:-2,:]
    colors = threecolors[:,0]
    bins = np.arange(0,xmax+0.015,0.01)

    xes = copy.deepcopy(points[:,0])

    points[:,0] = bins[np.digitize(xes,bins,right=True)]

    redptsinlastbins = np.count_nonzero(colors[np.where(points[:,0]== bins[-1])])+np.count_nonzero(colors[np.where(points[:,0]== bins[-2])])
    redptsinfirstbins = np.count_nonzero(colors[np.where(points[:,0]== bins[0])])+np.count_nonzero(colors[np.where(points[:,0]== bins[1])])
    
    if redptsinfirstbins > redptsinlastbins:
        return True
    else:
        return False

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


def getFeatureNewPad(pis,maxLayers):
    features = np.reshape(pis[0],(1,1024))
    for l in range(1,maxLayers):
        if l in pis:
            features = np.concatenate((features, np.reshape(pis[l],(1,1024))),axis=1)
        else:
            features = np.concatenate((features, np.ones((1,1024))),axis=1)

    return features

def mygetCamPosViewingDirection(pcd):
    points = np.asarray(pcd.points)
    cam2tr = points[-1,:]
    cam1tr = points[-2,:]
    direction = cam2tr-cam1tr
    unitvec = direction/np.linalg.norm(direction)
    return cam2tr, unitvec       
    
def getFacesToKeep(selectedface,facecounts):
    keep = [selectedface]
    if selectedface > 2:
        notkeep = [selectedface - 3]
    else:
        notkeep = [selectedface + 3]
    
    for f in range(3):
        if (f not in keep) and (f not in notkeep):
            if facecounts[f] > facecounts[f+3]:
                keep.append(f)
                notkeep.append(f+3)
            else:
                notkeep.append(f)
                keep.append(f+3)
    return keep,notkeep

def facearea(extent,faceid):
    if faceid ==0 or faceid==3:
        area =  extent[1]*extent[2]
    elif faceid== 1 or faceid==4:
        area = extent[0]*extent[2]
    else:
        area = extent[1]*extent[0]   
    return area
        
def getSelectedFaceAreas(extent,keep):
    areas = []
    for f in keep:
        areas.append(facearea(extent,f))
    return areas

def getFlowFace(faces,f):
    curveop = getCurvatureKDTreeFaceWGrad(faces[f],f)
    flow = getGradRep(curveop[1])[3]
    return flow

def getSelectedFlow(faces,keep):
    flow = []
    for f in keep:
        if len(faces[f])>0:
            flow.append(getFlowFace(faces,f))
        else:
            flow.append(9999)
    return flow


def compareArea2(a1,a2,percent):
    
    ## percent is the similarity percent (to be considered same the actually bigger value must be within percent % of smaller value)
    
    ##return codes
    ## 0 for same
    ## 1 if first is bigger
    ## 2 if second is bigger
    smalla = min(a1,a2)
    biga = max(a1,a2)
    
    diffa = biga-smalla
    if diffa <= percent*0.01*biga: #conservative
        #print('same')
        return 0
    else:
        #print(biga, 'is bigger')
        if biga == a1:
            return 1
        else:
            return 2

def compareArea3(a1,a2,a3,percent):
    
    ## percent is the similarity percent (to be considered same the actually bigger value must be within percent % of smaller value)
    
    ##return codes
    ## 0 for all same
    ## 11 if first two same, third smaller 12 if first two same, third greater
    ## 21 if last two same, first smaller 22 if last two same, first greater
    ## 31 first and last same, second smaller 32 if first and last same, second greater
    ## 4 all different
    
    onetwo = compareArea2(a1,a2,percent)
    twothree = compareArea2(a2,a3,percent)
    threeone = compareArea2(a3,a1,percent)    
    if onetwo == 0: #first two are similar
        if twothree == 0 and threeone == 0: #if third is similar to either
            #print('all same')
            return 0
        else:
            #print('first two same, third different')
            if twothree == 2 and threeone == 1:
                return 12
            elif twothree == 1 and threeone == 2:
                return 11
            else:
                if a3 < min(a1,a2):
                    return 11
                else:
                    return 12
                
            
            
    elif twothree == 0: #last two are similar
        if onetwo == 0 and threeone == 0: #if first is similar to either
            #print('all same')
            return 0
        else:
            #print('last two same, first different')
            if onetwo == 2 and threeone == 1:
                return 21
            elif onetwo == 1 and threeone == 2:
                return 22
            else:
                if a1 < min(a2,a3):
                    return 21
                else:
                    return 22
                    
            
    elif threeone == 0: #first and last are similar
        if onetwo == 0 and twothree == 0: #if first is similar to either
            print('all same')
            return 0
        else:
            #print('first and last same, second different')
            if onetwo == 1 and twothree == 2:
                return 31
            elif onetwo == 2 and twothree == 1:
                return 32
            else:
                if a2 < min(a1,a3):
                    return 31
                else:
                    return 32
            
    else:
        #print('all three different')
        return 4





def chooseModel(rc, keep, faces, selectedareas, percent):
    #first face in selected areas is selected face
    if rc == 4:
        #all three different
        if selectedareas[0] == max(selectedareas):
            return ['front','side']
        elif selectedareas[0] == min(selectedareas):
            return ['side','top']
        else:
            #print('need curvature')
            selectedflows = getSelectedFlow(faces,keep)
            if selectedflows[0] == min(selectedflows):
                return ['side','top']
            else:
                return ['front','side']
    elif rc == 0:
        #all three same 
        print('doesnt matter which i choose so go by front')
        return ['front']
    else:
        if rc == 11:
            return ['front','side']
        elif rc == 31:
            return ['front','side']
        elif rc == 12:
            return ['side','top']
        elif rc == 32:
            return ['side','top']
        elif rc == 21:
            return ['top']
        elif rc == 22:
            return ['front']
        else:
            print('error')
            return []




                    
#function to select pred from multiple layers
def selectLayerForPred(needtoflip,zs,maxLayers):
    if needtoflip:
        layerid = len(zs) - 1
    else:
        layerid = maxLayers - 1
        #layerid = len(zs)+1
    return layerid


def orientCamBottom(pcd):
    campos = np.asarray(pcd.points)[-1,:] 
    if campos[2] > 0:
        Rtemp = o3d.geometry.get_rotation_matrix_from_xyz([np.pi,0,0])
        pcd.rotate(Rtemp)
    return pcd
        
def filterDgm(dgm,thresh):
    newdgm = []
    for i in range(len(dgm)):
        if dgm[i,1] - dgm[i,0] <= thresh:
            newdgm.append((dgm[i,0],dgm[i,1],0))
    return np.asarray(newdgm)


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

def getRedColoredPoints(pcdorg,objptcolors):
    pts = np.asarray(pcdorg.points)
    return pts[np.where(objptcolors[:,0]== 1)], objptcolors[np.where(objptcolors[:,0]== 1)]

def separateRedBlackPoints(pcdorg,objptcolors):
    pts = np.asarray(pcdorg.points)
    return pts[np.where(objptcolors[:,0]== 1)], objptcolors[np.where(objptcolors[:,0]== 1)], pts[np.where(objptcolors[:,0]!= 1)], objptcolors[np.where(objptcolors[:,0]!= 1)] 

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

def maintest1(videodir,environment,category,separation,light):   
    if not os.path.exists('./temp/'):
        os.mkdir('./temp/')   
    if not os.path.exists('./temp/topsfeatures/'):
        os.mkdir('./temp/topsfeatures/')
    if not os.path.exists('./temp/savedinjson/'):
        os.mkdir('./temp/savedinjson/')
    if not os.path.exists('./temp/deptherrorlog/'):
        os.mkdir('./temp/deptherrorlog/')
        
    w = 1280
    h = 720
    fx = 920.458618164062
    fy = 921.807800292969
    ppx = 626.486083984375
    ppy = 358.848205566406
    cx = ppx
    cy = ppy
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx,fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic

    intvideolist = returnvideolist(environment,category,separation,light)
    
    videolist = [str(x) for x in intvideolist]

    maxLayers = 7
    refnormals = np.asarray([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])
  

    for video in videolist:
        if not os.path.exists(videodir +video+'/temp/'):
            os.mkdir(videodir +video+'/temp/')
        if not os.path.exists('./temp/topsfeatures/'+video):
            os.mkdir('./temp/topsfeatures/'+video)
        if not os.path.exists('./temp/savedinjson/'+video):
            os.mkdir('./temp/savedinjson/'+video)	
        if not os.path.exists('./temp/deptherrorlog/'+video):
            os.mkdir('./temp/deptherrorlog/'+video)	
        
        instances = sorted(fnmatch.filter(os.listdir(videodir +video+'/images/'),'*.yaml'))
        for i in instances:
            print(i, ' in ', video)

            fileid = i.split('_')[0]
            label = cv2.imread(videodir +video+'/images/'+fileid+'_labels.png',0)
            rgb = o3d.io.read_image(videodir +video+'/images/'+fileid+'_rgb.png')
            groundtruthdict = getGroundTruthFromYAML(videodir +video+'/images/'+fileid+'_poses.yaml')
            labelidxes = [i for i in np.unique(label) if i > 0]
            
            for idx in labelidxes:
                if groundtruthdict[idx] not in ['ice_cream','hot_sauce','chips_can']:
                    try: 
                        depthnp = cv2.imread(videodir +video+'/images/'+fileid+'_depth.png',-1)
                        contour = findContour(idx,label)
                        boundary = checkOccludeeContour(contour,idx,label,depthnp) ### depth with all objects here
                        
                        ## now get the depth of just the object of interest and create its point cloud
                        depthnp[np.where(label!=idx)] = 0
                        depthnp[np.where(depthnp > 2500)] = 0
                        depthnp[np.where(depthnp < 260)] = 0
                        cv2.imwrite(videodir +video+'/temp/'+fileid+'_depth_idx_'+str(idx)+'.png',depthnp)
                        
                        ##finding bad pixels inside object i.e. where depth is 0
                        depthnp[np.where(label!=idx)] = -1 ###note depth is of type uint16 ....here it actually gets assigned to 65535 but i dont care
                        badidxes = np.where(depthnp == 0)
                        
                        objptcolors = getObjectOcclusionColors(badidxes,idx,label,boundary) #red color points in object pcd are occlusion boundary

                        depth = o3d.io.read_image(videodir +video+'/temp/'+fileid+'_depth_idx_'+str(idx)+'.png')
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,depth_trunc = 2.5)
                        pcdorg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.intrinsic)#,cam.extrinsic)
                        pcdorg.colors = o3d.utility.Vector3dVector(objptcolors)


                        redpts, redcolors, blackpts,blackcolors = separateRedBlackPoints(pcdorg, objptcolors)
                        
                        pcdtodown = o3d.geometry.PointCloud()
                        pcdtodown.points = o3d.utility.Vector3dVector(blackpts)
                        downpcd = pcdtodown.voxel_down_sample(voxel_size=0.003)
                        
                        ##add red points later, not here so that they don't go away in outlier removal
                        
                        #t1 = time.time()
                        blackdownpcd,ind = downpcd.remove_radius_outlier(nb_points=220,radius=0.05)
                        #print(time.time() - t1)
                        
                        downpcdpts = np.asarray(blackdownpcd.points)
                        downpcdcolors = np.zeros_like(downpcdpts)
                        downpcdptsplusred = np.concatenate((downpcdpts,redpts),axis=0)
                        downpcdcolorsplusred = np.concatenate((downpcdcolors,redcolors),axis=0)
                        objpcd = o3d.geometry.PointCloud()
                        objpcd.points = o3d.utility.Vector3dVector(downpcdptsplusred)
                        objpcd.colors = o3d.utility.Vector3dVector(downpcdcolorsplusred)   

                            
                        pts = np.asarray(objpcd.points)
                        cam1 = np.expand_dims(np.asarray([0,0,-0.1]),axis=0)
                        cam2 = np.expand_dims(np.asarray([0,0,0]),axis=0) #test time cam position is 0,0,0
                        pts = np.concatenate((pts,cam1,cam2),axis=0)
                        colors = np.asarray(objpcd.colors)
                        camcol = np.expand_dims(np.asarray([0,0,0]),axis=0)
                        colors = np.concatenate((colors,camcol,camcol),axis=0)
                        #need to do the above to maintain red color of occluded points 
                        downpcdCam = o3d.geometry.PointCloud()
                        downpcdCam.points = o3d.utility.Vector3dVector(pts)
                        downpcdCam.colors = o3d.utility.Vector3dVector(colors)

    
                        rotatedpcd,extent = rotateToFlatForLayering(downpcdCam) 
                        scaledpcd = scaleObjectPCD(rotatedpcd,2.5) #scaling will scale cam positions appropriately   
        
                        needtoflip = checkNeedToFlipMinusCam(scaledpcd)
                        if needtoflip:
                            Rflip = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi])
                            scaledpcd.rotate(Rflip)
                        else:
                            donothing=1

                        trpcd = trXMinusCam(trYMinusCam(trZMinusCam(scaledpcd)))
                        rotatedpcd = orientCamBottom(trpcd)
        
                        flatpcd = copy.deepcopy(rotatedpcd)
                        
                        R45 = o3d.geometry.get_rotation_matrix_from_xyz([0,-np.pi/4,0])
                        rotatedpcd.rotate(R45)    

        
                        finalpcd = trXMinusCam(trYMinusCam(trZMinusCam(rotatedpcd)))
                        
        
            
                        pcdpts = np.asarray(finalpcd.points)[:-2,:]
                        rounded = roundinXYZ(pcdpts)
                        zs = getZs(rounded)
                        
        
                        pis = {}
                        for key,value in zs.items():
                            layer = getLayer(rounded,zs,key)
                            dgm = filterDgm(computePDBinningNo2DTranslation(layer),0.75)
                            img = pimgr.transform([dgm[:,0:2]]) 
                            pis[key] = img[0]
        

                        feature= getFeatureNewPad(pis,maxLayers)
                        
                        trflatpcd = trXMinusCam(trYMinusCam(trZMinusCam(flatpcd)))
                        trflatpcd.estimate_normals()
                        campos, viewingdirection = mygetCamPosViewingDirection(trflatpcd)
                        trflatpcd.orient_normals_towards_camera_location(campos)
         
                        normals = np.asarray(trflatpcd.normals)
                                
                        pointnormals = normals[:-2,:]
                        normals[-2,:] = viewingdirection #for checking in visualization..setting normal as viewing direction
                        normals[-1,:] = viewingdirection
                        trflatpcd.normals = o3d.utility.Vector3dVector(normals)
                        pcdpts = np.asarray(trflatpcd.points)[:-2,:]

                        
                        labels,faces,facecounts = clusteringNoCurvature(pointnormals,pcdpts,refnormals)
                        selectedface = 9
                        minfaceang = np.pi
                        
                        for fid in range(6):
                            normal = refnormals[fid]
                            
                            ang = np.pi - np.arccos(np.sum(normal*viewingdirection))
                            if ang < minfaceang:
                                minfaceang = ang
                                selectedface = fid
        
                        
                        keep,notkeep = getFacesToKeep(selectedface,facecounts)
        

                        newpcd = o3d.geometry.PointCloud()
                        newpcd.points = o3d.utility.Vector3dVector(pcdpts)
                        axis_aligned_bounding_box = newpcd.get_axis_aligned_bounding_box()
                        extent = axis_aligned_bounding_box.get_extent()  
                        selectedareas = getSelectedFaceAreas(extent,keep)

                        percent = 20
                        returncode = compareArea3(selectedareas[0],selectedareas[1],selectedareas[2],percent)

                        modeltypes = chooseModel(returncode,keep,faces,selectedareas,percent)
        
        
                        l = selectLayerForPred(needtoflip,zs,maxLayers)            
                        np.save('./temp/topsfeatures/'+video+'/'+ fileid+'_idx'+str(idx)+'.npy',np.nan_to_num(feature))

                        

                        return_dict = {}
                        return_dict['l'] = l
                        return_dict['modeltypes'] = modeltypes
                        return_dict['selectedareas'] = selectedareas
                        return_dict['returncode'] = returncode
                        
                        
                        with open('./temp/savedinjson/'+video+'/'+fileid+'_idx'+str(idx)+'.json','w') as write:
                            json.dump(return_dict,write)
                        

                    except:
                        with open('./temp/deptherrorlog/'+video+'/'+fileid+'_idx'+str(idx)+'#'+groundtruthdict[idx]+'.txt', 'w') as f:
                            f.write('error')
                        print('Depth sensing error occured in ', i, ' for idx ', idx, ' i.e., ', groundtruthdict[idx])
                    
       
    
        shutil.rmtree(videodir +video+'/temp/')   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videodir')
    parser.add_argument('--environment')
    parser.add_argument('--category')
    parser.add_argument('--separation')
    parser.add_argument('--light')   
    args = parser.parse_args()
    maintest1(args.videodir,args.environment,args.category,args.separation,args.light)
        
        

