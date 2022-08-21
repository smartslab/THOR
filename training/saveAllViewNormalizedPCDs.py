#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Ekta Samani
"""


import cv2
import numpy as np
import open3d as o3d
import os,argparse
import copy 

def scaleObjectPCD(pcd,scaleFactor):
    scaled = copy.deepcopy(pcd)
    scaled.scale(scaleFactor,center=[0,0,0])
    return scaled

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

def rotateToFlatForLayering(pcd):
    pcdpts = np.asarray(pcd.points)[:-2,:]
    bbox = o3d.geometry.OrientedBoundingBox()
    bboxresult = bbox.create_from_points(o3d.utility.Vector3dVector(pcdpts))#,robust=True)
    #o3d.visualization.draw([pcd,bboxresult])
    Rnew = np.transpose(bboxresult.R)
    pcd.rotate(Rnew)
    #now 2d using opencv


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
    #cv2.imwrite('myplot.png',final)
    kernel = np.ones((11, 11), np.uint8)
      
    # Using cv2.erode() method 
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


def main(data_path):        

    cam_a = [i for i in range(0,360,5)]
    cam_b = [i for i in range(0,185,5)]
    cam_a_remove = []
    cam_b_remove = [0,5,175,180]     
    cam_a_final = list(set(cam_a) - set(cam_a_remove))
    cam_b_final = list(set(cam_b) - set(cam_b_remove))
                
                
    
    object_list = os.listdir(data_path)
    for oname in object_list:
        print(oname)

        bdeglist = cam_b_final
            
    
        for bdeg in bdeglist:
            folder = str(bdeg)+'/0'
            if not os.path.exists(data_path+oname+'/'+folder+'/'+'vnpcdwcam/'):
                os.makedirs(data_path+oname+'/'+folder+'/'+'vnpcdwcam/')
            for file in cam_a_final:
    
                objpcd = o3d.io.read_point_cloud(data_path+oname+'/'+folder+'/pcd/'+str(file)+'.pcd') ##all the pcds in  the folder must be at a depth scale of 0.001m
                
                ##note that in some cases point clouds from depth images are distorted and not generated to the desired depth scale of 0.001m. In such cases appropriate a scale factor
                ## is manually determined and used to ensure all pcds at a depth scale of 0.001 before performing the following scaling by a factor of 2.5
                
                scaled = scaleObjectPCD(objpcd,2.5) #choosen scale factor
    
                ## the 
                downpcd = scaled.voxel_down_sample(voxel_size=0.01)
                    
                #need cam to check campos in rotateforlayering...reducing the flipz augmentation
                pts = np.asarray(downpcd.points)
                cam1 = np.expand_dims(np.asarray([0,0,-0.1]),axis=0)
                cam2 = np.expand_dims(np.asarray([0,0,0]),axis=0)
                pts = np.concatenate((pts,cam1,cam2),axis=0)
    
                downpcdCam = o3d.geometry.PointCloud()
                downpcdCam.points = o3d.utility.Vector3dVector(pts)
    
    
                R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,-np.pi/2])
                downpcdCam.rotate(R)
                
                rotatedpcd,_ = rotateToFlatForLayering(downpcdCam)
                o3d.io.write_point_cloud(data_path+oname+'/'+folder+'/'+'vnpcdwcam/'+str(file)+'.pcd',rotatedpcd)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    main(args.data_path)

