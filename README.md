#  🚀 Prédiction CTR Multimodale sur MicroLens-1M 
## 📌 Aperçu du Projet

Ce dépôt contient la solution pour la **Compétition de Prédiction du Taux de Clics Multimodaux (CTR)** basée sur le jeu de données 
[MicroLens-1M](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/)

L'objectif de ce projet est de prédire la probabilité qu'un utilisateur clique sur un élément spécifique (vidéo/article) en se basant sur :
- **Historique Utilisateur :**  Comportement séquentiel (clics passés).
- **Contenu de l'Élément :** Caractéristiques multimodales (Titres Textuels + Couvertures Images).

Mon approche repose sur une **architecture d'Optimisation en Cascade.** Contrairement à une méthode 'end-to-end' souvent trop lourde, j’ai séparé l'extraction de connaissances (Task 1) de la modélisation comportementale (Task 2). Cette stratégie m’a permis d'utiliser des modèles de pointe comme CLIP tout en gardant un modèle de prédiction agile.

- **Task 1** : Extraction d'embeddings multimodaux avec CLIP
- **Task 2** : Modèle CTR avec architecture Attention + DNN
- **Meilleur AUC** : 0.7752 (validation)

## 🏗️ Architecture
1. **CLIP fine-tuning** : Fusion texte (titre) + image (couverture)
2. **Réduction PCA** : 1024D → 128D
3. **Modèle CTR** : User/Item embeddings + Multi-head Attention + Deep NN
<img width="1261" height="1364" alt="diagram-export-20-12-2025-12_45_52" src="https://github.com/user-attachments/assets/ca8ea7d4-4550-460c-906a-15632be4201f" />

## 📦 Installation
```bash
# Cloner le dépôt
git clone https://github.com/oumniya03/Projet_competition.git
cd Projet_competition

# Installer les dépendances
pip install torch torchvision transformers pandas numpy scikit-learn tqdm
```

**Prérequis** :
- Python 3.11+
- CUDA 11.8+ (recommandé pour GPU)
  
## 🚀 Utilisation
```bash
# Exécuter le notebook complet
jupyter notebook competition.ipynb
```

## 📊 Résultats
- AUC validation : **0.7752**
- Dataset : 3.6M interactions train, 91K items

## 🛠️ Stack Technique
- Deep Learning : PyTorch 2.0, Transformers (CLIP)
- Data Processing : Pandas, NumPy
- ML Tools : scikit-learn (PCA), tqdm
- Compute : NVIDIA Tesla T4 (Kaggle)

## 👤 Author

**Oumniya Moutaouakil**
- Master's Student in Advanced Machine Learning & Multimedia Intelligence.
- GitHub: [@oumniya03](https://github.com/oumniya03)
- Project: [Projet-competition](https://github.com/oumniya03/Projet_competition.git)





