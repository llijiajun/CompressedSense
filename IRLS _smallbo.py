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

def cs_irls(y,T_Mat):   
    L=max(2,math.floor((y.shape[0])/4))
    hat_x_tp=np.dot(T_Mat.T ,y)
    epsilong=1
    p=1 # solution for l-norm p
    times=1
    while (epsilong>10e-6) and (times<L):  #迭代次数
        weight=(hat_x_tp**2+epsilong)**(p/2-1)
        Q_Mat=np.diag(1/weight)
        #hat_x=Q_Mat*T_Mat'*inv(T_Mat*Q_Mat*T_Mat')*y
        temp=np.dot(np.dot(T_Mat,Q_Mat),T_Mat.T)
        temp=np.dot(np.dot(Q_Mat,T_Mat.T),np.linalg.inv(temp))
        hat_x=np.dot(temp,y)
        if(np.linalg.norm(hat_x-hat_x_tp,2) < np.sqrt(epsilong)/100):
            epsilong = epsilong/10
        hat_x_tp=hat_x
        times=times+1
    return hat_x


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
        #print('正在重建第',i,'列。。。')
        column_rec=cs_irls(img_cs_1d[:,i],Theta_1d)  #利用sp算法计算稀疏系数
        sparse_rec_1d[:,i]=column_rec;        
    img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

    return img_rec

#读bmp
im=np.array(Image.open("lena512.bmp"))



#提取小波基
db1=pywt.Wavelet('db1')

#计算小波系数，最多可以分9层，可以分少一点
coef=pywt.wavedec(im,db1,level=9)

temp_coef=coef.copy()

fl=open("IRLS-smallbo-samplerate-rsquare.csv",'w')


coef=temp_coef.copy()
#comprass 0.1
coef[6]=comprase(coef[6],32,512,0.29)
coef[7]=comprase(coef[7],64,512,0.29*0.29)
coef[8]=comprase(coef[8],128,512,0.29*0.29*0.29)
coef[9]=comprase(coef[9],256,512,0.29*0.29*0.29*0.29)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.1,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.1.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.2
coef[7]=comprase(coef[7],64,512,0.3)
coef[8]=comprase(coef[8],128,512,0.3*0.3)
coef[9]=comprase(coef[9],256,512,0.3*0.3*0.3)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.2,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.2.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.3
coef[8]=comprase(coef[8],128,512,0.155)
coef[9]=comprase(coef[9],256,512,0.155*0.155)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.3,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.3.jpg',newimg)


coef=temp_coef.copy()
#comprass 0.4
coef[8]=comprase(coef[8],128,512,0.356)
coef[9]=comprase(coef[9],256,512,0.356*0.356)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.4,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.4.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.5
coef[9]=comprase(coef[9],256,512,0.004)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.5,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.5.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.6
coef[9]=comprase(coef[9],256,512,52.2/256)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.6,"+str(d2)+"\n")
cv2.imwrite('IRLS-smallbo/IRLS_smallbo_0.6.jpg',newimg)
fl.close()
