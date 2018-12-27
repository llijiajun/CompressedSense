# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import pywt
import math
from scipy.misc import imsave
from PIL import Image  #对应pillow包
import os
from numpy import *


#IHT算法函数
def cs_IHT(y,D):
    hat_x_tp=np.dot(D.T ,y)
    K=math.floor(y.shape[0]/3)  #稀疏度
    result_temp=np.zeros((512))  #初始化重建信号   
    u=0.5  #影响因子
    result=result_temp
    for j in range(K):  #迭代次数
        x_increase=np.dot(D.T,(y-np.dot(D,result_temp)))    #x=D*(y-D*y0)
        result=result_temp+np.dot(x_increase,u) #   x(t+1)=x(t)+D*(y-D*y0)
        temp=np.fabs(result)
        pos=temp.argsort() 
        pos=pos[::-1]#反向，得到前面L个大的位置
        result[pos[K:]]=0
        result_temp=result

    return  result


#读取图像，并变成numpy类型的 array
im=np.array(Image.open("lena512.bmp"))
hang=512
lie=512
sampleRate=1

damage=im.copy()
for i in range(60,90):
    for j in range(50,80):
        damage[i,j]=0

u, s, vh = np.linalg.svd(im)
#print(s.shape[0])
Phi = u[:int(s.shape[0]*sampleRate),]


#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((hang,hang))
v=range(hang)
for k in range(0,hang):  
    dct_1d=np.cos(np.dot(v,k*math.pi/hang))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)
#随机测量
#print(Phi.shape)
#print(image.shape)
dia=np.zeros([u.shape[0],vh.shape[0]])
for i in range(int(sampleRate*len(s))):
    dia[i,i]=s[i]
img_cs_1d=np.dot(Phi,dia)
img_cs_1d=np,dot(img_cs_1d,vh)
#np.dot(Phi,s)#np.dot(Phi,image)
img_cs_1d=img_cs_1d[1]
   
#重建
sparse_rec_1d=np.zeros((hang,lie))   # 初始化稀疏系数矩阵    
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(lie):
    #print('正在重建第',i,'列。。。')
    column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;        
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
dama=np.dot(u,img_rec)

image2=Image.fromarray(dama)
image2.show()
