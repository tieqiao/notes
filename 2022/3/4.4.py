

from __future__ import print_function
import types 
import cv2
import os
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from numpy.linalg import solve, norm
import open3d as o3d

def camera():
    # filename="/nfs/hpc/share/wangtie/2022/2/21/2D/v1/camera_output_DevEv_S_12_01_MobileInfants_trim.npy"
    filename="/nfs/hpc/share/wangtie/2022/2/21/2D/v3/camera_output-DevEv_S_12_01_MobileInfants_trim.npy"
    data=np.load(filename, allow_pickle=True)
    c=[]
    K=[]
    for key in data.item():
        a=data.item()[key]
        c.append(a['T'])
        K.append(a['K'])
    c=np.array(c)
    K=np.array(K)
    return c,K

def checkpoints(mkpts,pts):
    a=np.mean(pts, axis=0)
    if np.linalg.norm(a-mkpts)<10:
        return 1
    else:
        return 0

def to_3D(points, M, c):
    P = []
    for p in points:
        p1 = [p[0],p[1],1.0]
        X = np.dot(np.linalg.pinv(M),p1)
        X = X / X[3]
        #XX  = np.copy(X)
        #XX[1] = X[2]; XX[2] = X[1]; XX[2] = -XX[2]

        xvec = np.copy(X)
        xvec[0] = c[0]-xvec[0]
        xvec[1] = c[1]-xvec[1]
        xvec[2] = c[2]-xvec[2]
        xvec = - xvec
        P.append(xvec[:3])
    return np.array(P)

def line_intersect(pt1,U1,pt2,U2):
    P = []

    for u1, u2 in zip(U1, U2):
        u1 = u1 / norm(u1)
        u2 = u2 / norm(u2)
        n = np.cross(u1, u2)
        n /= norm(n)

        n1 = np.cross(u1, n)
        n2 = np.cross(u2, n)

        t1 = pt1 + u1 * (np.dot((pt2 - pt1), n2) / np.dot(u1, n2))
        t2 = pt2 + u2 * (np.dot((pt1 - pt2), n1) / np.dot(u2, n1))
        p = (t1 + t2) / 2
        P.append(p)
    return np.array(P)


def pair():
    print("fdskljf")
    CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170]]
    # CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    #             [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
    #             [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    # CocoColors=CocoColors[:17]
    # CocoColors.append([255, 0, 0])
    # CocoColors.append([255, 0, 0])
    CocoColors=np.array(CocoColors)/255.0
    

    c,K=camera()
    filename="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npy"
    # data=np.load(filename, allow_pickle=True)
    # keypoints0=data.item()['keypoints']
    data=np.load(filename, allow_pickle=True)
    # print(data['metadata'])
    keypoints=data.item()['keypoints']
    # mkpts1=keypoints[0][1][0]
    # mkpts2=keypoints[0][2][0]
    # mkpts1=np.mean(mkpts1, axis=0)
    # mkpts2=np.mean(mkpts2, axis=0)
    # print(mkpts1)
    # print(mkpts2)
    # return
    # lst_view1=[]
    # lst_view2=[]
    lst_idx=[]
    lst_P=[]
    limit=5000
    preposition=0
    for i in range(len(keypoints)):
        view0=0
        pts1=keypoints[i][0]
        for j in range(len(pts1)):
            # if checkpoints(mkpts1,pts1[j]):
            view0=pts1[j]

        view1=0
        pts1=keypoints[i][1]
        for j in range(len(pts1)):
            # if checkpoints(mkpts1,pts1[j])==0:
            view1=pts1[j]

        view2=0
        pts1=keypoints[i][2]
        for j in range(len(pts1)):
            # if checkpoints(mkpts2,pts1[j])==0:
            view2=pts1[j]

        # view3=0
        # pts1=keypoints[i][3]
        # for j in range(len(pts1)):
        #     if checkpoints(mkpts2,pts1[j]):
        #         view3=pts1[j]
        flag1=-1
        flag2=-1
        fview1=0
        fview2=0
        if isinstance(view1, np.ndarray) and isinstance(view2, np.ndarray):
            flag1=1
            flag2=2
            fview1=view1
            fview2=view2
        if flag1==-1 or flag2==-1:
            continue
        # print(type(view1),type(view3))
        # continue
        # print(np.mean(view1, axis=0))
        # print(np.mean(view2, axis=0))
        
        # view1=view1[:,:2]
        # view2=view2[:,:2]
        # print(view1)
        # print(view2)
        U1 = to_3D(fview1, K[flag1], c[flag1])
        U2 = to_3D(fview2, K[flag2], c[flag2])
        P = line_intersect(c[flag1],U1,c[flag2],U2)
        # print(type(P),P.shape)
        curposition=np.mean(P,axis=0)
        if isinstance(preposition, int):
            preposition=np.mean(P,axis=0)
        else:
            dist=np.linalg.norm(preposition-curposition)
            if dist>5:continue
            preposition=curposition

            print(i,dist)
        # print(np.mean(P,axis=0))
        # return 
        # P=P.tolist()
        # P.append(c[1])
        # P.append(c[2])
        lst_P.append(P)
        lst_idx.append(i)
        if i%1000==0:
            print(i)
        if len(lst_P)>limit:
            break
    print(len(lst_P))
    print(len(lst_idx))
    # print(lst_idx)
    data={}
    for i in range(len(lst_idx)):
        data[lst_idx[i]]=lst_P[i]
        # print(lst_P[i].shape)
        # break
    np.save('data_4.4d_DevEv_S_12_01.npy', data)
pair()


