# Projet d'Entraînement avec VGG16

Ce projet a pour objectif de mettre en œuvre un modèle de réseau de neurones convolutionnel (CNN) utilisant l'architecture VGG16 pour la détection d'objets. Le modèle est entraîné sur un ensemble de données comprenant des images et des annotations XML associées.

## Architecture de VGG16

VGG16 est un modèle de réseau de neurones convolutif très populaire, connu pour son architecture profonde. Voici un aperçu de l'architecture utilisée dans ce projet :

Input: 224x224x3 image

VGG16

[Flatten]

[Dense 128]
[Dense 128]
[Dense 64]
[Dense 4 (sigmoid)]


## Fonctionnalités principales du projet

- Lecture des données à partir de fichiers XML annotés.
- Prétraitement des images pour les adapter à l'entrée du modèle VGG16.
- Entraînement du modèle avec les données d'entraînement et évaluation de la performance sur l'ensemble de test.

## Dépendances et Installation

Ce projet nécessite l'installation des bibliothèques suivantes :
- NumPy
- OpenCV
- Matplotlib
- Keras
- TensorFlow
- Scikit-learn
- Seaborn

## Images

<img src="https://github.com/Ismaz999/plate-number/blob/main/resultats_predictions/image_0.png" alt="Description de l'image 1" width="300"/>
<img src="https://github.com/Ismaz999/plate-number/blob/main/resultats_predictions/image_33.png" alt="Description de l'image 2" width="300"/>
<img src="https://github.com/Ismaz999/plate-number/blob/main/resultats_predictions/image_16.png" alt="Description de l'image 3" width="300"/>
<img src="https://github.com/Ismaz999/plate-number/blob/main/resultats_predictions/image_2.png" alt="Description de l'image 4" width="300"/>

## Résultats

### Graphiques de perte et de précision

![Graphique de perte](https://github.com/Ismaz999/plate-number/blob/main/loss.png)
![Graphique de précision](https://github.com/Ismaz999/plate-number/blob/main/accuracy.png)

### Matrice de confusion

![Matrice de confusion](https://github.com/Ismaz999/plate-number/blob/main/confusion%20matrix%20vgg16.png)

