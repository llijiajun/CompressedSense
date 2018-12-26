"""
IHT iterative hard thresholding
迭代硬阈值
"""
import time
import math
from scipy.misc import imsave
# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包


#读取图像，并变成numpy类型的 array
im = np.array(Image.open('lena512damage.bmp'))#图片大小256*256

#生成高斯随机测量矩阵
sampleRate=0.5  #采样率
Phi=np.random.randn(512,512)
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

resu=[0 for i in range(200)]


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

        resu[j]+=np.linalg.norm(result-hat_x_tp,2)
    return  result


time_start=time.time()

#重建
sparse_rec_1d=np.zeros((512,512))   # 初始化稀疏系数矩阵    
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(512):
    print('正在重建第',i,'列。。。')
    column_rec=cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;        
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()

time_end=time.time()
print('totally cost',time_end-time_start)


