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

    #SP算法函数
    def cs_sp(y,D):
        hat_x_tp=np.dot(D.T ,y) #D.T？
    
        K=math.floor(y.shape[0]/3)  
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((hang))

        product=np.fabs(np.dot(D.T,y))
        pos_temp=product.argsort()  #数组值从小到大的索引值
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos_current=pos_temp[0:K]#初始化索引集 对应初始化步骤1
        residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#初始化残差 对应初始化步骤2


    
        #while True:  #迭代次数
        for j in range(K):
        
            product=np.fabs(np.dot(D.T,residual_current))       
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            pos=np.union1d(pos_current,pos_temp[0:K])#对应步骤1     
            pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#对应步骤2  
            pos_temp=pos_temp[::-1]
            pos_last=pos_temp[0:K]#对应步骤3    
            residual_last=y-np.dot(D[:,pos_last],np.dot(np.linalg.pinv(D[:,pos_last]),y))#更新残差 #对应步骤4
            if np.linalg.norm(residual_last)>=np.linalg.norm(residual_current): #对应步骤5  
                pos_last=pos_current
                break
            residual_current=residual_last
            pos_current=pos_last
        
            result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #对应输出步骤

        result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #对应输出步骤  
        
        return  result
    #重建
    sparse_rec_1d=np.zeros((hang,lie))   # 初始化稀疏系数矩阵    
    Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
    for i in range(lie):
        #print('正在重建第',i,'列。。。')
        column_rec=cs_sp(img_cs_1d[:,i],Theta_1d)  #利用sp算法计算稀疏系数
        sparse_rec_1d[:,i]=column_rec;        
    img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

    return img_rec

begin=time.time()
#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena512.bmp'))#图片大小512*512

#提取小波基
db1=pywt.Wavelet('db1')

#计算小波系数，最多可以分9层，可以分少一点
coef=pywt.wavedec(im,db1,level=9)
temp_coef=coef.copy()

fl=open("SP-smallbo-samplerate-rsquare.csv",'w')

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
cv2.imwrite('SP-smallbo/SP_smallbo_0.1.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.2
coef[7]=comprase(coef[7],64,512,0.3)
coef[8]=comprase(coef[8],128,512,0.3*0.3)
coef[9]=comprase(coef[9],256,512,0.3*0.3*0.3)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.2,"+str(d2)+"\n")
cv2.imwrite('SP-smallbo/SP_smallbo_0.2.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.3
coef[8]=comprase(coef[8],128,512,0.155)
coef[9]=comprase(coef[9],256,512,0.155*0.155)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.3,"+str(d2)+"\n")
cv2.imwrite('SP-smallbo/SP_smallbo_0.3.jpg',newimg)


coef=temp_coef.copy()
#comprass 0.4
coef[8]=comprase(coef[8],128,512,0.356)
coef[9]=comprase(coef[9],256,512,0.356*0.356)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.4,"+str(d2)+"\n")
cv2.imwrite('SP-smallbo/SP_smallbo_0.4.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.5
coef[9]=comprase(coef[9],256,512,0.004)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.5,"+str(d2)+"\n")
cv2.imwrite('SP-smallbo/SP_smallbo_0.5.jpg',newimg)

coef=temp_coef.copy()
#comprass 0.6
coef[9]=comprase(coef[9],256,512,52.2/256)
newimg=pywt.waverec(coef,db1)
d1=im-newimg
d2=sum(multiply(d1,d1))/(512*512)
fl.write("0.6,"+str(d2)+"\n")
cv2.imwrite('SP-smallbo/SP_smallbo_0.6.jpg',newimg)
fl.close()

end=time.time()
print(end-begin)
