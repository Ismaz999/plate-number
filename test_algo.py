import cv2
import numpy as np
from scipy.ndimage import convolve

def convolve_moi(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    output = np.zeros(image.shape)

    for i in range(image_height):
        for j in range(image_width):
            
            sub_matrix = padded_image[i:i+kernel_height, j:j+kernel_width]


            output[i,j] = np.sum(kernel * sub_matrix)

    return output

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
            output = np.median(sub_matrix)
    return output