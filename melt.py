import cv2
import numpy as np
A=cv2.imread("melt.jpg")
B=cv2.imread('lena.jpg')
#generate Gaussian pyramid for A
level=6
temp=A.copy()
gpA=[temp]
restA=[]
for i in np.arange(level):
    dst=cv2.pyrDown(temp)
    gpA.append(dst)
    restA.append(cv2.subtract(temp,cv2.pyrUp(dst)))
    temp=dst.copy()

temp=B.copy()
gpB=[temp]
restB=[]
for i in np.arange(level):
    dst=cv2.pyrDown(temp)
    gpB.append(dst)
    restB.append(cv2.subtract(temp,cv2.pyrUp(dst)))
    temp=dst.copy()

# generate Laplacian Pyramid for A
lpA=[gpA[level-1]]
for i in np.arange(level-1,0,-1):
    GE=cv2.pyrUp(gpA[i])
    L=cv2.subtract(gpA[i-1],GE)
    lpA.append(L)
lpB=[gpB[level-1]]
for i in np.arange(level-1,0,-1):
    GE=cv2.pyrUp(gpB[i])
    L=cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = (la+lb)/2#np.hstack((la[:,::2], lb[:,1::2]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
    LS.append(ls)
# now reconstruct

ls_ = LS[0]   #这里LS[0]为高斯金字塔的最小图片
for i in range(1,level):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(LS[i],ls_)
ls_=cv2.add(cv2.add(restA[0],restB[0])/2,ls_)
real = (A+B)/2#np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)

cv2.imwrite('Direct_blending.jpg',real)
