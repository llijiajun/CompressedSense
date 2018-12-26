import pywt
import math
import time
# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
from scipy.misc import imsave
import cv2

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

    #IHT算法函数
    def cs_IHT(y,D):
        hat_x_tp=np.dot(D.T ,y)
        K=math.floor(y.shape[0]/3)  #稀疏度
        result_temp=np.zeros((hang))  #初始化重建信号   
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



    #重建
    sparse_rec_1d=np.zeros((hang,lie))   # 初始化稀疏系数矩阵    
    Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
    for i in range(lie):
        #print('正在重建第',i,'列。。。')
        column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
        sparse_rec_1d[:,i]=column_rec;        
    img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

    return img_rec

#读bmp文件
im=np.array(Image.open('lena512.bmp'))

"""
#读.jpg
img = cv2.imread("lena512damage.jpg")
#转一层
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""

#提取小波基
db1=pywt.Wavelet('db1')

#计算小波系数，最多可以分9层，可以分少一点
coef=pywt.wavedec(im,db1,level=9)

#提取第8.9层系数
im=coef[9]
imm=coef[8]

#第8.9层压缩重塑
coef[8]=comprase(imm,128,512,0.01)
coef[9]=comprase(im,256,512,0.01)

#还原图片
newimg=pywt.waverec(coef,db1)

#显示重建后的图片
image2=Image.fromarray(newimg)
image2.show()

#写到本地
cv2.imwrite('result.jpg',newimg)
