import matplotlib.pyplot as plt
from scipy import fftpack as fft
import scipy
import cv2
import cv2
import random
import os
import numpy as np
import time
import copy



def devisor(a):
    x32 = -32
    y32 = -32
    d = []
    while x32 < len(a[0]) - 33:
        
        x32 = x32 + 32
        y32 = -32
        while y32 < len(a) - 33:
            
            y32 = y32 + 32
            f = []
            for i in range(x32, x32 + 32):
                f.append([])
                for j in range(y32, y32 + 32):
                    f[i - x32].append([])
                    #if y32 > 300:
                     #   print(x32, ' ',y32)
                    try:
                        f[i - x32][j - y32].append(a[i][j][0])
                        f[i - x32][j - y32].append(a[i][j][1])
                        f[i - x32][j - y32].append(a[i][j][2])
                        
                    except:
                        p = 0
            
            f = np.array(f)
            #f = f.astype(np.uint8)
            if f.dtype == np.uint8:
                d.append(f)
    return d
a = cv2.imread('21.jpg')
w = devisor(a)
#f = cv2.resize(w[1], (100,100))


def get_data_mat(dir):
    X = []
    y = []
    print(dir)
    files = os.listdir(dir)
    
    random.shuffle(files)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv2.imread(os.path.join(dir, file))
            

            
            # cv2.imshow("ali", img)
            '''if file[:3] == 'fac':'''
            #     ****  X.append(noisy("poisson", img))   ****
            X.append(img)
            y.append(img)
    return np.array(X), np.array(y)
def merge(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])
    random.shuffle(c)
    return c








