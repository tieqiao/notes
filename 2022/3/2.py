

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


def fun_keypoints2(ppath,filename0,video_path0):
    #filename="/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_MobileInfants_trim1.npz"
    filename=ppath+filename0
    #filename="/nfs/hpc/share/wangtie/2022/2/21/2D/data_2d_DevEv_S_12_01_BottomLeft_trim.npz"
    data=np.load(filename, allow_pickle=True)
    keypoints=data.item()['keypoints']
    print(len(keypoints))

    video_path=ppath+video_path0

    vidcap = cv2.VideoCapture(video_path)


    name = video_path.split("/")[-1]
    base_name = name.split(".")[0]
    

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    save_path = 'output2_' + name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter(save_path,fourcc, 12.0, (int(vidcap.get(3)//2),int(vidcap.get(4)//2)))
    out = cv2.VideoWriter(save_path,fourcc, fps, (int(vidcap.get(3)),int(vidcap.get(4))))

    i=0
    delta=[]
    while True:
        ret, image_bgr = vidcap.read()
        
        # print(h,w)
        if i==0:
            h, w, _ = image_bgr.shape
            delta=np.array([[0,0],[w/2.0,0],[0,h/2.0],[w/2.0,h/2.0]])
            # print(delta)
        # return
        
        if ret:
            # for i in range(len(keypoints)):
                #print(i)
                

           
            # print(len(pts1))
            for j in range(4):
                pts1=keypoints[i][j]
                if len(pts1)>0:
                    # print(pts1[0].shape)
                    kpt=pts1[0][:,:2]
                    
                    kpt[:,0]=kpt[:,0]+delta[j][0]
                    kpt[:,1]=kpt[:,1]+delta[j][1]
                    # print(j,kpt)
                    # print(kpt[:,0])
                    # print(image_bgr.shape)
                    draw_pose(kpt,image_bgr)
            # return
        else:
            print('cannot load the video.')
            break
        out.write(image_bgr)
        i+=1
        print(i)
        # if i>30:
        #     break
    # print(cnt)
    vidcap.release()
    print('video has been saved as {}'.format(save_path))
    out.release()
    print('yes')


    # print(type(data))
#"/nfs/hpc/share/wangtie/2022/2/21/data_2d_DevEv_S_12_01_BottomLeft_trim1.npy"
#"/nfs/hpc/share/wangtie/2022/2/21/DevEv_S_12_01_BottomLeft_trim.mp4"
#fun_keypoints2("/nfs/hpc/share/wangtie/2022/2/21/","data_2d_DevEv_S_12_01_MobileInfants_trim1.npy","DevEv_S_12_01_MobileInfants_trim.mp4")
fun_keypoints2("/nfs/hpc/share/wangtie/2022/2/21/","data_2d_DevEv_S_12_01_BottomLeft_trim1.npy","DevEv_S_12_01_BottomLeft_trim.mp4")