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

def noisyf(a, p):
    for i in range(len(a)):
        for j in range(len(a[i])):
            e = random.randint(1, 100)
            if e <= p:
                for k in range(3):
                    a[i][j][k] = random.randint(0, 255)
def fourier(image, coef):
        
    fourier = fft.fft2(image)
    mag = np.abs(fourier)
    phase = np.angle(fourier)
    noise = np.random.random(image.shape) * coef * np.pi
    phase += noise
    fourier = mag * np.cos(phase) + mag * np.sin(phase) * 1j
    img_noise = fft.ifft2(fourier)
    img_noise = abs(img_noise)
    img_noise = 255 * ((img_noise - img_noise.min()) / (img_noise.max() - img_noise.min()))
    img_noise = img_noise.astype(np.uint8)
    return img_noise
def noisy(noise_typ, image, **par):
    if noise_typ == "gauss":
        try:
            var = par["var"]
        except:
            var = 500
        row,col,ch= image.shape
        mean = 0
        var = var
        sigma = var ** 0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        print(noisy.max(), noisy.min())
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        print(noisy.max(), noisy.min())
        return noisy
    elif noise_typ == "fourier":
        try:
            coef = par["coef"]
        except:
            coef = 0.4
        imgf = image.copy()
        imgf[:, :, 0] = fourier(imgf[:, :, 0], coef)
        imgf[:, :, 1] = fourier(imgf[:, :, 1], coef)
        imgf[:, :, 2] = fourier(imgf[:, :, 2], coef)
        return imgf
    elif noise_typ == "poisson":
        try:
            summit = par["summit"]
        except:
            summit = 225
        
        noise_mask = np.random.poisson(summit, size = image.shape) - summit
        noisy_img = image + noise_mask
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_img = noisy_img.astype(np.uint8)
        
        return noisy_img
    '''elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy'''

'''
filename = 'pic2.jpg'
img = (cv2.imread(filename))
img = cv2.resize(img, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_NEAREST)
guass_noise = noisy("gauss", img, var = 1000)
poisson_noise = noisy("poisson", img, summit = 500)
fourier_noise = noisy("fourier", img, coef = 0.7)
cv2.imshow("asli", img)
cv2.imshow("guass", guass_noise)
cv2.imshow("poisson", poisson_noise)
cv2.imshow("fourier", fourier_noise)
noisyf(img, 1)
cv2.imshow("s&p", img)

cv2.waitKey()
cv2.destroyAllWindows()
'''
'''for i in range(1000):
    a.append([])
    for j in range(1000):
        a[i].append([])
        a[i][j].append(i / 4)
        a[i][j].append(j / 4)
        a[i][j].append(1000 / 4 - i / 4)'''


'''a = cv2.resize(a, (600, 600), interpolation = cv2.INTER_NEAREST)
img = cv2.resize(a, (600, 600), interpolation = cv2.INTER_NEAREST)'''
'''imgr = []
imgg = []
imgb = []
for i in range(len(img)):
    imgr.append([])
    imgg.append([])
    imgb.append([])
    for j in range(len(img[i])):
        imgr[i].append(img[i][j][0])
        imgg[i].append(img[i][j][1])
        imgb[i].append(img[i][j][2])'''

'''z = np.zeros(img.shape[:2], dtype = np.uint8)
cv2.imshow("imgr", cv2.merge((img[:, :, 0], z, z)))
cv2.imshow("imgg", cv2.merge((z, img[:, :, 1], z)))
cv2.imshow("imgb", cv2.merge((z, z, img[:, :, 2])))

img[:, :, 0] = noisy("fourier", img[:, :, 0])
img[:, :, 1] = noisy("fourier", img[:, :, 1])
img[:, :, 2] = noisy("fourier", img[:, :, 2])'''



'''cv2.imshow("noisy", img)'''
'Parameters'
'----------'
'''image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
'''


