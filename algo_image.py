import cv2
import numpy as np
from scipy.ndimage import convolve
from test_algo import convolve_moi, noyau_gauss, filter_gauss, filtre_median, erosion_grayscale, dilatation_grayscale, ouverture_grayscale
import os

dossier_image = 'Kazakhstan\\files\\domain1\\900900.jpg'
chemin_image = os.path.join(os.getcwd(), dossier_image)

image = cv2.imread(chemin_image)
image_gris = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)

kernel_taille = 3 #nombre impair seulement
sigma = 1
kernel_eros = np.ones((kernel_taille, kernel_taille), np.uint8)

if image is not None:
    print("L'image a été chargée avec succès.")
    # cv2.imshow("Image Chargée", image) 
    # cv2.waitKey(0)
else:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")

np_sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
np_sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

image_gauss = filter_gauss(image_gris, kernel_taille, sigma)

image_gauss_uint = image_gauss.astype(np.uint8) #on convertit l'image en uint8 sinon elle ne s'affihe pas

########################################################################################
#####                           TEST DES ALGO                                      #####

#test de ma fonction de convolution pour le filtre de sobel
# fct_sobelx = convolve_moi(image_gris,np_sobelx)
# fct_sobely = convolve_moi(image_gris,np_sobely)

# #fonction interne de numpy sobel filter
# cnv_sobelx = convolve(image_gris, np_sobelx)
# cnv_sobely = convolve(image_gris, np_sobely)

# ####Filtre de Sobel
# sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# # Filtre de Canny
# edges = cv2.Canny(image, 100, 200)

# # Transformation de Hough pour la détection des lignes
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

# #FILTRE MEDIAN
# image_median = filtre_median(image_gris, kernel_taille)
# image_median_uint = image_median.astype(np.uint8)

#erosion
erosion_img = erosion_grayscale(image_gris, kernel_eros, threshold=90)
dilatation_img = dilatation_grayscale(image_gris, kernel_eros)
ouverture_img = ouverture_grayscale(image_gris, kernel_eros)

cv2.imshow('Image Originale', image)
# cv2.imshow('utilisatin gauss', image_gauss_uint)
# cv2.imshow('Sobel X', sobelx)
# cv2.imshow('filtre median de merde', image_median_uint)
cv2.imshow('erosion test', erosion_img)
cv2.imshow('dilat test', dilatation_img)
cv2.imshow('ouverture', ouverture_img)    
# cv2.imshow('Sobel Y', sobely)
# cv2.imshow('Sobel X scipy', cnv_sobelx)
# cv2.imshow('Sobel Y scipy', cnv_sobely)
# cv2.imshow('Sobel X fonction', cnv_sobelx)
# cv2.imshow('Sobel Y fonction', cnv_sobely)
# cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

######## I - Pré-traitement des images ############

###Détection de contours
## Sobel, Canny Edge, Hough transform
##
###Filtrage
## Filtre gaussien, medians
##
###Morphologie
## Erosion, Dilatation, Ouverture
##
###Normalisation
#####

######## II - Détection et segmentation des images ############

#####
## YOLOV5
## Réseau CNN
#####

######## III - Classification et reconnaissance des caractères ############

#####
## OCR (reconnaissance optique)
## Réseau CNN
#####