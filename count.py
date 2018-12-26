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

def count(name1,name2):
    im1 = np.array(Image.open(name1))
    img = cv2.imread(name2)
    im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d1=im1-im2
    d2=sum(multiply(d1,d1))/(512*512)
    return d2
name1='lena512.bmp'
name2="IHT_smallbo_0.5.jpg"
print(count(name1,name2))
