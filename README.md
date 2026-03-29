


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
dans le code:  
from dotenv import load_dotenv  
import os  
load_dotenv()  
api_key = os.getenv("MISTRAL_API_KEY")  
client = MistralClient(api_key=api_key)  
  
on peut crée aussi un fichier .env.example à commiter dans Git:  
ajouter les noms de variables mais sans les valeurs comme:  
MISTRAL_API_KEY=your_api_key_here

