import cv2
import numpy as np
from scipy.ndimage import convolve
from test_algo import convolve_moi, noyau_gauss, filter_gauss, filtre_median, erosion_grayscale, dilatation_grayscale, ouverture_grayscale
import os

# dossier_image = 'Kazakhstan\\files\\domain1\\900900.jpg'
# chemin_image = os.path.join(os.getcwd(), dossier_image)

# image = cv2.imread(chemin_image)
# image_gris = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)

kernel_taille = 3 #nombre impair seulement
sigma = 1
kernel_eros = np.ones((kernel_taille, kernel_taille), np.uint8)

chemin_kazak = 'KAZAK_test\\images'
chemin_sortie = 'KAZAK_test\\images_filtrées'

if not os.path.exists(chemin_sortie):
    os.makedirs(chemin_sortie)

for fichier in os.listdir(chemin_kazak):
    if fichier.lower().endswith(('.png','.jpg','.jpeg')):
        chemin_image = os.path.join(chemin_kazak, fichier)
        image = cv2.imread(chemin_image)

        if image is None:
            print(f"Erreur : Impossible de charger l'image {fichier}. Vérifiez le chemin.")
            continue

        image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_gauss = filter_gauss(image_gris, kernel_taille, sigma)
        erosion_img = erosion_grayscale(image_gris, kernel_eros, threshold=90)
        dilatation_img = dilatation_grayscale(image_gris, kernel_eros)
        ouverture_img = ouverture_grayscale(image_gris, kernel_eros)

        cv2.imwrite(os.path.join(chemin_sortie, f'gauss_{fichier}'), image_gauss)
        cv2.imwrite(os.path.join(chemin_sortie, f'erosion_{fichier}'), erosion_img)
        cv2.imwrite(os.path.join(chemin_sortie, f'dilatation_{fichier}'), dilatation_img)
        cv2.imwrite(os.path.join(chemin_sortie, f'ouverture_{fichier}'), ouverture_img)
    else:
        print(f"Erreur : Impossible de charger l'image {fichier}. Vérifiez le chemin.")

if image is not None:
    print("L'image a été chargée avec succès.")
    # cv2.imshow("Image Chargée", image) 
    # cv2.waitKey(0)
else:
    print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")

np_sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
np_sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])



image_gauss_uint = image_gauss.astype(np.uint8) #on convertit l'image en uint8 sinon elle ne s'affihe pas


