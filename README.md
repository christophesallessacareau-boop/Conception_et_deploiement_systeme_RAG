# Objectif:  
Déploiement d'un système RAG, intégrant des modèles Mistral avec LangChain  
  
# Fonctionnalités:  
Recherche sémantique avec FAISS pour trouver les documents  
Génération de réponses avec modèles Mistral

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
docker build -t rag-occitanie  
### Lancer le conteneur en injectant les clés depuis .env
docker run -p 8000:8000 --env-file .env rag-occitanie      
  

# FAST API:  

## lancer API:  
uvicorn api:app --reload  
  
## health check:  
curl http://localhost:8000/
  
## docs Sawagger (auto-générée):  
http://127.0.0.1:8000/docs  
  
## Test HTTP:  
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "Quels événements culturels à Toulouse ?"}'  
  
 ## Rebuild (avec la clé ADMIN):  
 curl -X POST "http://127.0.0.1:8000/rebuild" \  
-H "x-api-key: ADMIN_KEY"  
  
    
      

  









