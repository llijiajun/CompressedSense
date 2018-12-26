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
        column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
        sparse_rec_1d[:,i]=column_rec;        
    img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
    return img_rec

begin=time.time()
#读bmp
im=np.array(Image.open("lena512.bmp"))
#提取小波基
db1=pywt.Wavelet('db1')

#计算小波系数，最多可以分9层，可以分少一点
coef=pywt.wavedec(im,db1,level=9)
temp_coef=coef.copy()

fl=open("IHT-smallbo-samplerate-rsquare.csv",'w')

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
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.1.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.2
coef[7]=comprase(coef[7],64,512,0.3)
coef[8]=comprase(coef[8],128,512,0.3*0.3)
coef[9]=comprase(coef[9],256,512,0.3*0.3*0.3)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.2,"+str(d2)+"\n")
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.2.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.3
coef[8]=comprase(coef[8],128,512,0.155)
coef[9]=comprase(coef[9],256,512,0.155*0.155)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.3,"+str(d2)+"\n")
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.3.jpg',newimg)


coef=temp_coef.copy()
#comprass 0.4
coef[8]=comprase(coef[8],128,512,0.356)
coef[9]=comprase(coef[9],256,512,0.356*0.356)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.4,"+str(d2)+"\n")
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.4.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.5
coef[9]=comprase(coef[9],256,512,0.004)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.5,"+str(d2)+"\n")
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.5.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.6
coef[9]=comprase(coef[9],256,512,52.2/256)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.6,"+str(d2)+"\n")
cv2.imwrite('IHT-smallbo/IHT_smallbo_0.6.jpg',newimg)
fl.close()
end=time.time()
print("cost",end-begin)
