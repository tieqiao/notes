

from __future__ import print_function
from re import L
import types 
import cv2
import os
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from numpy.linalg import solve, norm

import pandas as pd
import open3d as o3d

def test(data,mdata,idx):
    v = 0.5
    a = 0.5
    sp= 0.1 # Sigma for position noise

    kalman = cv2.KalmanFilter(9, 3, 0)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, v, 0, 0, a, 0, 0],
        [0, 1, 0, 0, v, 0, 0, a, 0],
        [0, 0, 1, 0, 0, v, 0, 0, a]
    ],np.float32)

    kalman.transitionMatrix = np.array([
            [1, 0, 0, v, 0, 0, a, 0, 0],
            [0, 1, 0, 0, v, 0, 0, a, 0],
            [0, 0, 1, 0, 0, v, 0, 0, a],
            [0, 0, 0, 1, 0, 0, v, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, v, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, v],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],np.float32)

    kalman.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ],np.float32) * 0.007

    kalman.measurementNoiseCov = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0 ,1]
        ],np.float32) * sp

    kalman.statePre = np.array([
                            [np.float32(data[0][idx][0])], [np.float32(data[0][idx][1])], [np.float32(data[0][idx][2])]
                            , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
                            , [np.float32(0.)], [np.float32(0.)], [np.float32(0.)]
                            ])
    mp = np.array((3, 1), np.float32) # measurement
    tp = np.zeros((3, 1), np.float32)

    for i in range(data.shape[0]):
        mp = np.array([
            [np.float32(data[i][idx][0])],
            [np.float32(data[i][idx][1])],
            [np.float32(data[i][idx][2])],
        ])
        kalman.correct(mp)
        tp = kalman.predict()
        for j in range(3):
            mdata[i][idx][j]=float(tp[j])



def pair():
    
    # filename="/nfs/hpc/share/wangtie/2022/1/17/2D/data_2d_DevEv_S_12_01_MobileInfants_trim.npz"
    filename="/nfs/hpc/share/wangtie/2022/2/24/data_5.4d_DevEv_S_12_01.npy"
    data=np.load(filename, allow_pickle=True)
    lst=[]
    for key in data.item():
        lst.append(np.array(data.item()[key]))
        ##break
    data=np.array(lst)
    
    data3=[]
    for i in range(data.shape[0]):
        if i==0 or i==data.shape[0]-1:
            data3.append(data[i])
        else:
            data3.append(np.mean(data[i-1:i+1],axis=0))

    data2={}
    lenlen=int(len(data3)/2)
    for i in range(lenlen):
        data2[i]=data3[i*2]
        # print(data[i])
        # print(lst_P[i].shape)
        # break
    print(lenlen)
    np.save('data_5.5d_DevEv_S_12_01.npy', data2)
        # return 
    # for i in range(140,177):
    #     # print(i)
    #     print(i,data[i][0])
    # mdata=np.zeros_like(data)
    # for i in range(17):
    #     print(i)
    #     test(data,mdata,i)

    # kdata={}
    # for i in range(data.shape[0]):
    #     kdata[i]=mdata[i]
    #     # print(lst_P[i].shape)
    #     # break
    # np.save('data_4d_DevEv_S_12_01_MobileInfants_trim.npy', kdata)
    # print(data[100][10])
    # print(mdata[100][10])
    # print(mdata.shape)
pair()


