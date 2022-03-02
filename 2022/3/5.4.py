

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
    filename="/nfs/hpc/share/wangtie/2022/2/24/data_5.3d_DevEv_S_12_01.npy"
    data=np.load(filename, allow_pickle=True)
    # data=np.array(data.item())
    # print(data.shape)
    # return 
    # print(type(data))
    # print(data.shape)
    lst=[]
    for key in data.item():
        lst.append(np.array(data.item()[key]))
        ##break
    data=np.array(lst)

    print(data.shape)
    PreP=np.mean(data[0],axis=0)

    frame_idx=[]
    delta=[]
    delta_value=[]
    for i in range(data.shape[0]):
        P=np.mean(data[i],axis=0)
        # print(i,P-PreP)
        deltaP=P-PreP
        deltaP_norm=np.linalg.norm(deltaP)
        # if deltaP_norm>0.5:
        #     data[i]=data[i-1]
        #     deltaP=[-1,-1,-1]
        #     deltaP_norm=-1
        delta.append(deltaP)
        frame_idx.append(i)
        delta_value.append(deltaP_norm)

        PreP=P
    delta_value.sort()
    idx=len(delta_value)*0.95
    idx=int(idx)
    # print(idx)
    threshold=delta_value[idx]
    print(idx,threshold)

    frame_idx=[]
    delta=[]
    delta_value=[]
    PreP=np.mean(data[0],axis=0)
    data2=data.copy()

    cnt=0
    for i in range(data.shape[0]):
        P=np.mean(data[i],axis=0)
        # print(i,P-PreP)
        deltaP=P-PreP
        deltaP_norm=np.linalg.norm(deltaP)
        if deltaP_norm>threshold:
            cnt+=1
            if cnt>70:
                print(i)
                tdata=np.mean(data2[i:i+30],axis=0)
                # print(tdata.shape)
                data[i]=tdata
                deltaP=[-2,-2,-2]
                deltaP_norm=-2
                P=np.mean(data[i],axis=0)
                cnt=0
                # return 
            else:
                data[i]=data[i-1]
                deltaP=[-1,-1,-1]
                deltaP_norm=-1
                P=np.mean(data[i],axis=0)
        else:
            cnt=0
        delta.append(deltaP)
        frame_idx.append(i)
        delta_value.append(deltaP_norm)
        PreP=P

    # frame_idx=[]
    # delta=[]
    # delta_value=[]
    # PreP=np.mean(data[4999],axis=0)
    # for i in range(data.shape[0]-1,-1,-1):
    #     P=np.mean(data[i],axis=0)
    #     # print(i,P-PreP)
    #     deltaP=P-PreP
    #     deltaP_norm=np.linalg.norm(deltaP)
    #     if deltaP_norm>threshold:
    #         data[i]=data[i+1]
    #         deltaP=[-1,-1,-1]
    #         deltaP_norm=-1
    #         # P=np.mean(data[i],axis=0)
    #     delta.append(deltaP)
    #     frame_idx.append(i)
    #     delta_value.append(deltaP_norm)
    #     PreP=P
    # # return

    # frame_idx=[]
    # delta=[]
    # delta_value=[]
    # PreP=np.mean(data[0],axis=0)

    # for i in range(data.shape[0]):
    #     P=np.mean(data[i],axis=0)
    #     # print(i,P-PreP)
    #     deltaP=P-PreP
    #     deltaP_norm=np.linalg.norm(deltaP)
    #     if deltaP_norm>threshold:
    #         data[i]=data[i-1]
    #         deltaP=[-1,-1,-1]
    #         deltaP_norm=-1
    #         # P=np.mean(data[i],axis=0)
    #     delta.append(deltaP)
    #     frame_idx.append(i)
    #     delta_value.append(deltaP_norm)
    #     PreP=P
    
    dataframe = pd.DataFrame({'frame_idx':frame_idx,'delta':delta,'delta_value':delta_value})
    dataframe.to_csv("results5.4.csv",index=False,sep=',')

    data2={}
    for i in range(data.shape[0]):
        data2[i]=data[i]
        # print(data[i])
        # print(lst_P[i].shape)
        # break
    np.save('data_5.4d_DevEv_S_12_01.npy', data2)
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


