# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 21:09:01 2018

@author: 89217
"""

import cv2
import numpy as np
import time
import math
from scipy.misc import imsave
from PIL import Image  #对应pillow包
import os

resu=[0 for i in range(200)]

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

        resu[j]+=np.linalg.norm(result-hat_x_tp,2)
    return  result


def comprase(image,lie,hang,sampleRate):
    #sampleRate=0.01  #采样率
    Phi=np.random.randn(hang,hang)
    u, s, vh = np.linalg.svd(Phi)
    Phi = u[:int(hang*sampleRate),] #将测量矩阵正交化


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
    img_cs_1d=np.dot(Phi,image)

    
    #重建
    sparse_rec_1d=np.zeros((hang,lie))   # 初始化稀疏系数矩阵    
    Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
    for i in range(lie):
        print('正在重建第',i,'列。。。')
        column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
        sparse_rec_1d[:,i]=column_rec;        
    img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
    return img_rec

#读bmp
img=np.array(Image.open("lena512.bmp"))

#laplace pyramid 的提取
pyramid_img=[]
rest=[]
level=2
temp=img.copy()
for i in range(level):
    dst=cv2.pyrDown(temp)
    pyramid_img.append(dst)
    rest.append(cv2.subtract(temp,cv2.pyrUp(dst)))
    temp=dst.copy()
    
im=rest[0].copy()

#压缩最上面一层
rest[0]=comprase(im,512,512,0.05)


#重塑图像,可以考虑多压缩几层
begin=1
cancover=cv2.pyrUp(pyramid_img[begin])
for i in range(begin+1):
    lpls=rest[begin-i]+cancover
    cancover=cv2.pyrUp(lpls)

image2=Image.fromarray(lpls)
image2.show()
cv2.imwrite('IHT-lpls-0.3.jpg',lpls)
