import cv2
import numpy as np
from scipy.ndimage import convolve

chemin_image = 'C:\\Users\\Audensiel\\Desktop\\numero_plaque\\Kazakhstan\\files\\domain1\\900900.jpg'
image = cv2.imread(chemin_image)
image_gris = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)

# kernel = np.array()

if image is not None:
    print("L'image a été chargée avec succès.")
    cv2.imshow("Image Chargée", image)
    cv2.waitKey(0)
else:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")

################ OPEN CV ##################
####Filtre de Sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# Filtre de Canny
edges = cv2.Canny(image, 100, 200)

# Transformation de Hough pour la détection des lignes
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
############################################

np_sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
np_sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

cnv_sobelx = convolve(image_gris, np_sobelx)
cnv_sobely = convolve(image_gris, np_sobely)


cv2.imshow('Image Originale', image)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Sobel X scipy', cnv_sobelx)
cv2.imshow('Sobel Y scipy', cnv_sobely)
cv2.imshow('Canny', edges)
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