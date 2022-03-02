

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
from scipy.optimize import least_squares

import pandas as pd

def camera():
    filename1="/nfs/hpc/share/wangtie/2022/2/21/2D/camera_output_DevEv_S_12_01_MobileInfants_trim.npy"
    filename2="/nfs/hpc/share/wangtie/2022/2/21/2D/camera_output_DevEv_S_12_01_BottomLeft_trim.npy"
    c=[]
    K=[]
    data=np.load(filename1, allow_pickle=True)
    for key in data.item():
        #print(key)
        a=data.item()[key]
        c.append(a['T'])
        K.append(a['K'])

    data=np.load(filename2, allow_pickle=True)
    for key in data.item():
        #print(key)
        a=data.item()[key]
        c.append(a['T'])
        K.append(a['K'])

    
    c=np.array(c)
    K=np.array(K)
    return c,K

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        #if i == 14 : 
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def fun_rosenbrock(x,M,keypoints,zidx):
    a=[]
    A=[]
    cnt=0
    for j in range(8):
        if len(keypoints[j])>0:
            
            # print(keypoints[j][0][zidx].shape)
            # print(M[j].shape)
            # print(type(keypoints[j][0][zidx].shape))
            # print(type(M[j]))
            # print(keypoints[j][0][zidx])
            # print(j)
            a0=keypoints[j][0][zidx][:3]
            a0[2]=1
            M0=np.array(M[j])
            if cnt==0:
                a=a0
                A=M0
            else:
                a=np.concatenate((a, a0), axis=0)
                A=np.concatenate((A, M0), axis=0)
            cnt+=1
    
    # print(a)
    # print(cnt)
    # print(a.shape)
    # print(A.shape)
    # print(x[zidx].shape)
    x0 = np.expand_dims(x, axis=1)
    # print(x0.shape)
    # print(A[0])
    # print(x0)
    # print(a)
    residuals=np.matmul(A,x0).squeeze()-a
    # print(residuals.shape)
    return residuals

    print(residuals)
    # print(a)
    print(residuals.shape)

def fun_keypoints2(ppath):
    #filename="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npz"
    filename0="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npy"
    filename1="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_BottomLeft_trim1.npy"
    filename=filename0
    data=np.load(filename, allow_pickle=True)
    keypoints0=data.item()['keypoints']
    filename=filename1
    data=np.load(filename, allow_pickle=True)
    keypoints1=data.item()['keypoints']

    c,M=camera()
    # print(c.shape)
    # print(M.shape)

    ans=[]
    cost=[]
    frame_idx=[]
    for i in range(len(keypoints0)):
        frame_idx.append(i)
        init_ans=np.zeros((17,4))
        keypoints=keypoints0[i]
        keypoints.extend(keypoints1[i])
        if i!=0:
            init_ans=np.array(ans[i-1])

        cnt=0
        for j in range(8):
            if len(keypoints[j])>0:
                cnt+=1
        print(i)
        # if i>10:
        #     break
        if cnt<=1:
            ans.append(ans[i-1])
        else:
            # fun_rosenbrock(init_ans,M,keypoints,0)
            tans=[]
            tcost=0
            for j in range(17):
                x0_rosenbrock = init_ans[j]
                # print(x0_rosenbrock.shape)
                # fun_rosenbrock(x0_rosenbrock,M,keypoints,0)
                res_1 = least_squares(fun_rosenbrock,x0_rosenbrock,args=(M,keypoints,j))
                # print(res_1.x)
                # print(res_1.cost)
                a=res_1.x/res_1.x[3]
                # print(a[:3])
                tans.append(a)
                tcost+=res_1.cost
            # print(np.array(tans).shape)
            ans.append(np.array(tans))
            cost.append(tcost)
        dataframe = pd.DataFrame({'frame_idx':frame_idx,'cost':cost})
        dataframe.to_csv("results.csv",index=False,sep=',')
    print(len(ans))
    lenans=len(ans)
    ans=np.array(ans)
    ans=ans[:,:,:3]
    # for i in range(lenans):
    #     print(ans[i].shape)
    #     print(ans[])
    #     break
    # ans=np.array(ans)
    # print(ans.shape)
    
    data={}
    for i in range(len(ans)):
        data[i]=ans[i]
        # print(data[i])
        # print(lst_P[i].shape)
        # break
    np.save('data_3d_DevEv_S_12_01.npy', data)



            # return 
        # print(len(a))
        # print(type(a))

        # return 

def fun_rosenbrock(x,M,keypoints,zidx):
    a=[]
    A=[]
    cnt=0
    for j in range(8):
        if len(keypoints[j])>0:
            
            # print(keypoints[j][0][zidx].shape)
            # print(M[j].shape)
            # print(type(keypoints[j][0][zidx].shape))
            # print(type(M[j]))
            # print(keypoints[j][0][zidx])
            # print(j)
            a0=keypoints[j][0][zidx][:3]
            a0[2]=1
            M0=np.array(M[j])
            if cnt==0:
                a=a0
                A=M0
            else:
                a=np.concatenate((a, a0), axis=0)
                A=np.concatenate((A, M0), axis=0)
            cnt+=1
    
    # print(a)
    # print(cnt)
    # print(a.shape)
    # print(A.shape)
    # print(x[zidx].shape)
    x0 = np.expand_dims(x, axis=1)
    # print(x0.shape)
    # print(A[0])
    # print(x0)
    # print(a)
    residuals=np.matmul(A,x0).squeeze()-a
    # print(residuals.shape)
    return residuals

def fun_keypoints4(M):
    F=np.zeros((8,8,3,3))
    points4=[]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points4.append([i,j,k,1])
    points4=np.array(points4).T
    # print(M.shape)
    # print(points4.shape)
    # print(points4)
    points3=np.matmul(M,points4).transpose(0,2,1)
    print(points3.shape)
    for i in range(points3.shape[0]):
        for j in range(points3.shape[1]):
            points3[i][j]=points3[i][j]/points3[i][j][2]
    # points3=points3.astype(int)
    
    # points3=np.transpose()
    # print(points3.shape)
    # for i in range(8):
    #     points3=points3[i]/points3[]
    # print(points3[0].shape)
    # return
    print("No")
    for i in range(points3.shape[0]):
        for j in range(points3.shape[0]):
            # if i==0 and j==1:
            F1, mask=cv2.findFundamentalMat(points3[i],points3[j])
            F[i][j]=F1
                # rst=np.matmul(points3[j],np.matmul(F1,points3[i].T))
                # print(i,j,np.sum(rst))
                # return 
                # print(points3[i][0])
                # print(points3[j][0])
                # a=np.matmul(F1,np.expand_dims(points3[j][0], axis=1))
                # print(F1)
                # print(a)
    for i in range(points3.shape[0]):
        for j in range(points3.shape[0]):
            # if i==0 and j==5:
                rst=np.matmul(points3[j],np.matmul(F[i][j],points3[i].T))
                # print(rst.shape)
                print(i,j,np.sum(rst))
                # print(i,j,np.sum(rst))
                # return
                # print(F1)
                # print(type(F1))
                # return 
            


def fun_keypoints3(ppath):
    #filename="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npz"
    filename0="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npy"
    filename1="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_BottomLeft_trim1.npy"
    filename=filename0
    data=np.load(filename, allow_pickle=True)
    keypoints0=data.item()['keypoints']
    filename=filename1
    data=np.load(filename, allow_pickle=True)
    keypoints1=data.item()['keypoints']

    c,M=camera()
    fun_keypoints4(M)
    # print(c.shape)
    # print(M.shape)


fun_keypoints3("/nfs/hpc/share/wangtie/2022/2/21/")
