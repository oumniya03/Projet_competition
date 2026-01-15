#  Prédiction CTR Multimodale sur MicroLens-1M 
## 📌 Aperçu du Projet

Ce dépôt contient la solution pour la **Compétition de Prédiction du Taux de Clics Multimodaux (CTR)** basée sur le jeu de données **MicroLens-1M**.

L'objectif de ce projet est de prédire la probabilité qu'un utilisateur clique sur un élément spécifique (vidéo/article) en se basant sur :
- **Historique Utilisateur :**  Comportement séquentiel (clics passés).
- **Contenu de l'Élément :** Caractéristiques multimodales (Titres Textuels + Couvertures Images).

Notre approche met en œuvre un **Pipeline d'Optimisation en Cascade** combinant de Grands Modèles Vision-Langage (CLIP) pour l'extraction de caractéristiques et des Réseaux d'Intérêt Profond (DIN) pour la prédiction du CTR.

## 🏗️ Architecture & Méthodologie
La solution est divisée en trois tâches stratégiques :

### Tâche 1 : Apprentissage de Représentation Multimodale (Offline)

Au lieu d'utiliser des embeddings de base fixes, nous avons optimisé les représentations des éléments pour capturer de meilleurs signaux sémantiques.

- **Modèle :** openai/clip-vit-base-patch32 (Vision Transformer).
- **Technique :** Fine-tuning de CLIP sur les données d'interaction utilisateur-élément (Label 1 = Paire positive, Label 0 = Paire négative) pour aligner l'espace visuel/textuel avec les préférences des utilisateurs.
- **Réduction de Dimensionnalité :** Compression des embeddings 1280D (Texte+Image) vers 128D en utilisant une PCA/Autoencodeur pour correspondre aux exigences d'entrée du modèle CTR.
### 2. Tâche 2 : Optimisation du Modèle CTR (Online)

Nous avons utilisé des architectures de recommandation de pointe pour modéliser les intérêts des utilisateurs.

- **Modèle :** DIN (Deep Interest Network).
- **Pourquoi DIN ?** Il utilise un mécanisme d'attention pour activer uniquement les parties de l'historique utilisateur qui sont pertinentes pour l'élément candidat actuel.
- **Framework :** Implémenté avec FuxiCTR.
- **Réglage des Hyperparamètres :** Optimisation de la Taille du Batch, du Taux d'Apprentissage et du Dropout pour prévenir le Surapprentissage et les erreurs OOM (Out Of Memory) sur GPU.

### 3. Tâche 1 & 2 : Fusion en Cascade

La soumission finale combine les forces des deux tâches :

- **Étape A :** Générer des embeddings multimodaux optimisés en utilisant le CLIP fine-tuné (Tâche 1).

- **Étape B :** Injecter ces embeddings dans le modèle DIN (Tâche 2) en tant que vecteurs de caractéristiques denses.

- **Étape C :** Entraîner le modèle CTR de bout en bout pour exploiter les riches informations sémantiques.

## 🛠️ Architecture

<img width="1261" height="1364" alt="diagram-export-20-12-2025-12_45_52" src="https://github.com/user-attachments/assets/ca8ea7d4-4550-460c-906a-15632be4201f" />




## 📦 🛠️ Installation & Configuration


### 1. Clone the Repository

```bash
git clone https://github.com/oumniya03/Projet-rag.git
cd Projet-rag
```

### 2. Configure Environment Variables

For security reasons, API keys are not stored in the repository. You must create a `.env` file at the root of the project.

Create a file named `.env` and add your Groq API Key:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

> **Note:** You can obtain a Groq API key from [https://console.groq.com](https://console.groq.com)

### 3. Run with Docker Script

The project includes a helper script `docker.sh` as requested in the challenge requirements.

**Start the application:**

```bash
chmod +x docker.sh  # Only needed the first time (Linux/Mac/GitBash)
./docker.sh up
```

> **Note:** The first launch may take a few minutes to build the images and download the embedding model.

### 4. Access the Application

Once the containers are running, you can access:

* **Frontend (Chat UI):** [http://localhost:3000](http://localhost:3000)
* **Backend (API Docs):** [http://localhost:8000/docs](http://localhost:8000/docs)

## 🔧 Management Commands (docker.sh)

| Command | Description |
|---------|-------------|
| `./docker.sh up` | Builds and starts the containers in the background |
| `./docker.sh down` | Stops and removes containers and networks |
| `./docker.sh build` | Rebuilds the images (useful after code changes) |
| `./docker.sh logs` | Displays real-time logs from backend and frontend |


## 📂 Project Structure

```
Projet-rag/
├── backend/
│   ├── main.py              # FastAPI Application & RAG Logic
│   ├── Dockerfile           # Backend Container Configuration
│   └── requirements.txt     # Python Dependencies
├── frontend/
│   ├── app/
│   │   └── page.tsx         # Main Chat Interface (React/Next.js)
│   ├── Dockerfile           # Frontend Container Configuration
│   └── tailwind.config.ts   # Styling Configuration
├── data/                    # Persisted Data (Indices & Uploads)
├── docker-compose.yml       # Docker Orchestration
├── docker.sh                # Management Script
├── .env                     # Environment Variables (create this)
└── README.md                # This file
```



## 📝 Usage Guide

### Uploading Documents

1. Navigate to [http://localhost:3000](http://localhost:3000)
2. Use the upload interface to add your documents
3. Wait for the processing confirmation

### Asking Questions

1. Type your question in the chat interface
2. Press Enter or click Send
3. View the AI-generated answer with source references

### API Integration

You can integrate the backend API into your own applications:

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Your question here"}
)

answer = response.json()["answer"]
sources = response.json()["sources"]
```

## 🔍 Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use:

```bash
# Modify docker-compose.yml to use different ports
# Change "3000:3000" to "3001:3000" for frontend
# Change "8000:8000" to "8001:8000" for backend
```

### Container Build Failures

```bash
./docker.sh down
docker system prune -a
./docker.sh build
./docker.sh up
```

### Missing Environment Variables

Ensure your `.env` file exists and contains:
```env
GROQ_API_KEY=your_actual_key_here
```

## 🤝 Contributing

This is a challenge submission project. For questions or feedback, please contact the author.


## 👤 Author

**Oumniya**

- GitHub: [@oumniya03](https://github.com/oumniya03)
- Project: [Projet-rag](https://github.com/oumniya03/Projet-rag)

## 🎯 Challenge Requirements Met

- ✅ Dockerized application with `docker.sh` script
- ✅ RAG implementation with document retrieval
- ✅ `/ask` endpoint with proper response format
- ✅ Evaluation script for quality assessment
- ✅ Modern, responsive frontend interface
- ✅ Complete documentation

---



*Built  using FastAPI, Next.js, and Docker*

