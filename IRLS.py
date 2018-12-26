import math
import time
# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
from scipy.misc import imsave

#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena512.bmp'))
#print (im.shape, im.dtype)uint8

#生成高斯随机测量矩阵
sampleRate = 0.7 #采样率
Phi = np.random.randn(512, 512)
u, s, vh = np.linalg.svd(Phi)
Phi = u[:int(512*sampleRate),] #将测量矩阵正交化

#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((512,512))
v=range(512)
for k in range(0,512):  
    dct_1d=np.cos(np.dot(v,k*math.pi/512))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#随机测量
img_cs_1d=np.dot(Phi,im)
#print(len(img_cs_1d),len(img_cs_1d[0]))

#IRLS算法函数
def cs_irls(y,T_Mat):   
    L=math.floor((y.shape[0])/4)
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



#重建
sparse_rec_1d=np.zeros((512,512))   # 初始化稀疏系数矩阵    
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵

time_start=time.time()

for i in range(512):
    print('正在重建第',i,'列。。。')
    column_rec=cs_irls(img_cs_1d[:,i],Theta_1d)  #利用IRLS算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;        
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)

time_end=time.time()
print('totally cost',time_end-time_start)

#imsave("IRLS-"+str(rate)+".jpg",image2)
image2.show()

"""
imsave("yasuo.jpg",image2)
"""
