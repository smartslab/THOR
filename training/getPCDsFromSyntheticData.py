#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ekta Samani
"""
import open3d as o3d
import csv,os
import numpy as np
import ast,cv2
import matplotlib.pyplot as plt


object_list = os.listdir('./library/')

for obj_name in object_list:
    print(obj_name)
    for cam_b_deg in range(0,185,5):
        folder = str(cam_b_deg)+'/0'
        if not os.path.exists('./library/'+obj_name+'/'+folder+'/'+'pcd/'):
            os.makedirs('./library/'+obj_name+'/'+folder+'/'+'pcd/')
            
        intrinsics = {}
        #extrinsics = {}
        
        with open('./library/'+obj_name+'/'+folder+'/'+obj_name+'.csv', mode='r') as inp:
            reader = csv.reader(inp)
            intrinsics = {rows[0]:np.reshape(ast.literal_eval(rows[1]),(3,3)) for rows in reader}
        
        
        for k,v in intrinsics.items():
            f = 1099 #fov40
            depth_img = np.squeeze(np.load('./library/'+obj_name+'/'+folder+'/'+'depth/'+os.path.splitext(k)[0]+'.npy'))
            p2d_idx = np.where(depth_img>-1)
            us = p2d_idx[0]
            vs = p2d_idx[1]
            values = np.squeeze(depth_img[us,vs])
            p2d_value = np.vstack((us,vs,values)).T
            p3d = p2d_value.copy()
            p3d[:,0] = -(p3d[:,0]-300)/f#/p3d[:,2]
            p3d[:,1] = (p3d[:,1]-400)/f#/p3d[:,2]
            objpcd = p3d[np.where(p3d[:,2]< 1.0)]
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(objpcd)
            o3d.io.write_point_cloud('./library/'+obj_name+'/'+folder+'/'+'pcd/'+os.path.splitext(k)[0]+'.pcd', pcl)
