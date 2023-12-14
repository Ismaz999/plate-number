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
            output[i,j] = np.median(sub_matrix)
    return output

def erosion_grayscale(image, kernel, threshold=127):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    print("height", kernel_height)
    output = np.zeros_like(image)
    print("output image",output)

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    for i in range(image_height):
        for j in range(image_width):
            sub_matrix = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Appliquer l'érosion en considérant les valeurs supérieures à threshold comme 1
            output[i, j] = np.min(sub_matrix[kernel == 1]) > threshold
    print("iimage finie",output)
    return output * 255

def dilatation_grayscale(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Créer une image de sortie
    output = np.zeros_like(image)

    # Padding autour de l'image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    for i in range(image_height):
        for j in range(image_width):
            # Extraire la sous-matrice
            sub_matrix = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Appliquer la dilatation (maximum des valeurs sous le noyau)
            output[i, j] = np.max(sub_matrix)

    return output

def ouverture_grayscale(image, kernel):
    # Appliquer d'abord l'érosion
    eroded_image = erosion_grayscale(image, kernel)

    # Appliquer ensuite la dilatation sur l'image érodée
    opened_image = dilatation_grayscale(eroded_image, kernel)

    return opened_image