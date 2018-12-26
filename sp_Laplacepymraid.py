#-*-coding:utf-8 -*-

import cv2
import numpy as np
import time
import math
from scipy.misc import imsave
from PIL import Image  #对应pillow包
import os

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
        print('正在重建第',i,'列。。。')
        column_rec=cs_sp(img_cs_1d[:,i],Theta_1d)  #利用sp算法计算稀疏系数
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
cv2.imwrite("SP-lpls-0.3.jpg",lpls)
