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


