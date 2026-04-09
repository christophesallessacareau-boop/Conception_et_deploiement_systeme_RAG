
# Image de base 
# Python 3.11 sur une base légère (slim = sans outils inutiles)
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Dépendances système nécessaires pour faiss et numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*    # nettoyage pour réduire la taille de l'image

# Installation des dépendances Python 
# On copie requirements.txt EN PREMIER (optimisation cache Docker)
# Si le code change mais pas requirements.txt → cette étape n'est pas rejouée
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY api.py .
COPY RAG/ ./RAG/

# Port exposé par l'API FastAPI
EXPOSE 8000

# Commande de démarrage
# --host 0.0.0.0 → accessible depuis l'extérieur du conteneur
# --port 8000    → port exposé
# --reload       → à retirer en production
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]