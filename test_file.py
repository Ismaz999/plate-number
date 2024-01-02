import cv2
import numpy as np
from scipy.ndimage import convolve
from fonction_cv import convolve_moi, noyau_gauss, filter_gauss, filtre_median, erosion_grayscale, dilatation_grayscale, ouverture_grayscale
import os
import pandas as pd
from sklearn.model_selection import train_test_split

kernel_taille = 3 #nombre impair seulement
sigma = 1
kernel_eros = np.ones((kernel_taille, kernel_taille), np.uint8)

chemin_kazak = 'KAZAK_test\\images'
chemin_sortie = 'KAZAK_test\\filtered_images_test'
chemin_annot = 'KAZAK_test\\annot\\Kazakhstan_domain1_p1_samples.csv'

if not os.path.exists(chemin_sortie):
    os.makedirs(chemin_sortie)

for fichier in os.listdir(chemin_kazak):
    if fichier.lower().endswith(('.png','.jpg','.jpeg')):
        base_filename, file_extension = os.path.splitext(fichier)
        already_processed = True

        # Check if all filtered images exist
        for filter_type in ['gauss', 'erosion', 'dilatation', 'ouverture']:
            if not os.path.exists(os.path.join(chemin_sortie, f'{filter_type}_{fichier}')):
                already_processed = False
                break

        if already_processed:
            print(f"{fichier} has already been processed. Skipping...")
            continue

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


noms_colonnes = [
    'file_name', 'bbox', 'license_plate_id', 'license_plate_visibility',
    'license_plate_rows_count', 'license_plate_number', 'license_plate_serial',
    'license_plate_country', 'license_plate_color', 'license_plate_mask'
]

df_annotations = pd.read_csv(chemin_annot, names=noms_colonnes)
df_annotations = df_annotations.iloc[1:]
df_annotations = df_annotations.reset_index(drop=True)
print(df_annotations.head())

X = []
y = []

for _, row in df_annotations.iterrows():
    fichier_image = row['file_name']
    chemin_image = os.path.join(chemin_sortie, fichier_image)
    
    # Chargement de l'image
    image = cv2.imread(chemin_image)
    if image is not None:
        X.append(image)
        y.append(row['bbox'])  # Remplacer 'bbox' par le nom de la colonne appropriée

# Conversion de X et y en tableaux NumPy pour l'entraînement
X = np.array(X)
y = np.array(y)


print("Images chargées :", len(X))
print("Annotations chargées :", len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

