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


def comprase(image,lie,hang,sampleRate):
    #sampleRate=0.01  #采样率
    
    u, s, vh = np.linalg.svd(image)
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
    return np.dot(u,img_rec)

begin=time.time()
#读bmp
im=np.array(Image.open("lena512.bmp"))
#提取小波基

damage=im.copy()
for i in range(60,90):
    for j in range(50,80):
        damage[i,j]=0
        
image2=Image.fromarray(damage)
image2.show()
db1=pywt.Wavelet('db1')

#计算小波系数，最多可以分9层，可以分少一点
coef=pywt.wavedec(im,db1,level=9)
coef_damage=pywt.wavedec(damage,db1,level=9)

temp_coef=coef.copy()
coef=temp_coef.copy()
#comprass 0.6
#coef_damage[3]=comprase(coef_damage[3],4,512,1)
#coef_damage[4]=comprase(coef_damage[4],8,512,1)
#coef_damage[5]=comprase(coef_damage[5],16,512,1)
#coef_damage[6]=comprase(coef_damage[6],32,512,1)
#coef_damage[7]=comprase(coef_damage[7],64,512,1)
coef_damage[8]=comprase(coef_damage[8],128,512,1)
coef_damage[9]=comprase(coef_damage[9],256,512,1)
newimg=pywt.waverec(coef_damage,db1)

image2=Image.fromarray(newimg)
image2.show()

end=time.time()
print("cost",end-begin)
"""
fl=open("02.txt",'w')
fl2=open("03.txt",'w')
for i in range(len(coef)):
    for j in range(len(coef[i][0])):
        for k in range(len(coef[i])):
            v=fl.write(str(coef[i][k][j])+",")
        v=fl.write("\n")
for i in range(len(coef)):
    for j in range(len(coef[i][0])):
        for k in range(len(coef[i])):
            v=fl2.write(str(coef_damage[i][k][j])+",")
        v=fl2.write("\n")
fl.close()
fl2.close()
"""
