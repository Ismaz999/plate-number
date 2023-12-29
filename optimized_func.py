import cv2
import numpy as np
from scipy.ndimage import convolve, median_filter
from scipy import signal
from skimage.morphology import erosion, square, dilation, opening

def convolve_moi(image, kernel):
    return signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

def noyau_gauss(size, sigma=1):
    size = int(size)//2
    x,y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/(2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0 *sigma**2))) * normal
    print("ca cest g", g)
    return g

def filter_gauss(image, kernel_size, sigma):
    kernel_g = noyau_gauss(kernel_size, sigma)
    return convolve_moi(image, kernel_g)

def filtre_median(image, kernel):
    pad_size = kernel // 2
    output = np.zeros_like(image)
    image_pad = np.pad(image, pad_size, mode='constant')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sub_matrix = image_pad[i:i+kernel, j:j+kernel]
            output[i,j] = np.median(sub_matrix)
    return output

def erosion_grayscale(image, kernel_size):
    kernel = square(kernel_size)
    return erosion(image, kernel)

def dilatation_grayscale(image, kernel_size):
    kernel = square(kernel_size)
    return dilation(image, kernel)

def ouverture_grayscale(image, kernel_size):
    kernel = square(kernel_size)
    return opening(image, kernel)