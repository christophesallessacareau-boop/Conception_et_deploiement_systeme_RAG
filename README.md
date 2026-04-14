# Objectif:  
Déploiement d'un système RAG, intégrant des modèles Mistral avec LangChain  
  
# Fonctionnalités:  
Recherche sémantique avec FAISS pour trouver les documents  
Génération de réponses avec modèles Mistral  
  
  
[![Tests RAG](https://github.com/christophesallessacareau-boop/Concevez_et_d-ployez_un_syst-me_RAG/actions/workflows/test.yml/badge.svg)](https://github.com/christophesallessacareau-boop/Concevez_et_d-ployez_un_syst-me_RAG/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Vector Search](https://img.shields.io/badge/vector%20search-FAISS-purple)
![RAG](https://img.shields.io/badge/RAG-LangChain-orange)
![LLM](https://img.shields.io/badge/LLM-Mistral-red)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/docker-ready-blue)  
  

# Prérequis  
Python 3.11.9  
Clé API Mistral (obtenue sur console.mistral.ai)

# Cloner le Repo  
git clone https://github.com/christophesallessacareau-boop/Concevez_et_d-ployez_un_syst-me_RAG  
cd Concevez_et_d-ployez_un_syst-me_RAG

# Remarque: pour simuler un environnement propre:  
on vérifie que le requirements.txt est autosuffisant:

## Désactiver et supprimer le venv  
deactivate  
Remove-Item -Recurse -Force .venv  

## Recréer un venv vierge  
py -3.11 -m venv .venv  

## activer l'environnement  
.venv\Scripts\Activate.ps1  

## Réinstaller depuis le requirements.txt  
pip install -r requirements.txt  
  
## Vider le cache pip
pip cache purge  

# N.B: pour utiliser une clé avec API Mistral:  
pip install python-dotenv  
dans le fichier .env ou .venv (.env A RENSEIGNER dans .gitignore):  
MISTRAL_API_KEY=noter_la_clé_ici  
  

# DOCKER:  
# Commandes pour builder et lancer:  
## avec Docker Compose:  
### Builder et lancer en une commande  

docker compose up --build  

### En arrière-plan  

docker compose up --build -d  

### Arrêter  

docker compose down  

## ou bien avec Docker seul:  
### Builder l'image  
docker build -t rag-occitanie .  
### Lancer le conteneur en injectant les clés depuis .env
docker run -p 8000:8000 --env-file .env rag-occitanie      
  

# FAST API:  

## lancer API:  
### Lancement uvicorn avec port fixe
uvicorn api:app --reload --port 8000
  
## health check:  
curl http://localhost:8000/
  
## docs Sawagger (auto-générés):  
http://127.0.0.1:8000/docs  
  
## Test HTTP:  
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Quels événements culturels à Toulouse et Toulouse UNIQUEMENT en 2026 ?"}'  
  
 ## Rebuild (avec la clé ADMIN à renseigner manuellement):  
 curl -X POST http://127.0.0.1:8000/admin/rebuild \
     -H "X-ADMIN-Key: ADMIN_KEY"  
## ou bien Rebuild (avec clé ADMIN automatique):  
python admin_tools.py

  
    
      

  









