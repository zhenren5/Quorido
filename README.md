# Projet Quoridor IA

## Description

Ce projet vise à développer une intelligence artificielle (IA) en Python 3 pour le jeu **Quoridor**, un jeu de plateau stratégique à deux joueurs. L’objectif est de créer une IA capable d'apprendre à jouer efficacement en utilisant des méthodes d’apprentissage par renforcement combinées à des réseaux de neurones.

L'IA ne dispose d'aucune stratégie prédéfinie basée sur des connaissances humaines. Elle apprend uniquement en jouant des parties répétées contre elle-même.

## Règles du jeu

Le jeu **Quoridor** se joue sur un plateau de taille N×NN \times NN×N (par défaut N=9N = 9N=9). Chaque joueur dispose de MMM murs (par défaut M=10M = 10M=10). Les règles sont simples :

1.  Le but est d’atteindre en premier la rangée opposée du plateau.
2.  À chaque tour, un joueur peut :
    - Déplacer son pion sur une case adjacente (haut, bas, gauche, droite).
    - Placer un mur pour gêner son adversaire.
3.  Un joueur ne peut pas bloquer totalement l'accès de son adversaire à la rangée cible.
4.  Le pion peut sauter par-dessus l’adversaire sous certaines conditions.

## Fonctionnalités du projet

- **Plateau paramétrable** : Support de différentes tailles de plateau (N×NN \times NN×N) et nombre de murs (MMM).
- **Apprentissage par renforcement** :
  - **Stratégie Off-line** : Monte Carlo.
  - **Stratégies On-line** : TD(0) et Q-learning.
  - **Stratégie avancée** : TD(λ) utilisant des réseaux de neurones.
- **Interface utilisateur** : Conçue avec **PyQt5** pour permettre une interaction intuitive avec le jeu.

## Installation

1.  **Cloner le projet** :

        git clone https://github.com/zhenren5/Quorido.git

2.  **Installer les dépendances** :  
     Utilisez un environnement virtuel pour gérer les dépendances :

        python3 -m venv venv
        source venv/bin/activate
        # Sous Windows :
        venv\Scripts\activate

        pip install -r requirements.txt

3.  **Lancer le projet** :

    `python quorido.py`

## Screenshot du Jeu

Voici un aperçu de l'interface du jeu Quoridor :
![about page](/images/interface.png?raw=true "interface")
