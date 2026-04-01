
import sys
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import time
import requests
from datetime import datetime, timedelta, timezone

from mistralai import Mistral
print("OK import") # pour verification de l'import de mistral!!!!!!
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Récupération de la clé API Mistral depuis le fichier .env:

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)
print("Clé chargée :", MISTRAL_API_KEY is not None)


# Récupérer les événements via l'API OpenDataSoft

def telecharger_evenements():
    """Récupère les événements d'Occitanie via l'API OpenDataSoft de Openagenda."""
 
    # Date d'il y a 1 an (format attendu par l'API)
    il_y_a_un_an = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
 
    # Paramètres de la requête
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    params = {
        "where": f"location_region='Occitanie' AND firstdate_end >= date'{il_y_a_un_an}'",
        "limit": 100,           # pour 100 événements
        "order_by": "firstdate_begin DESC", # trier par date décroissante
    }
 
    print("Téléchargement des événements en cours...")
    reponse = requests.get(url, params=params)

    # afficher le statut pour détecter les erreurs silencieuses de l'API:
    print("Statut HTTP :", reponse.status_code)
    if reponse.status_code != 200:
        print(" Erreur API :", reponse.text[:500])
        return []
    
    donnees = reponse.json()
 
    evenements = donnees.get("results", [])
    print(f"{len(evenements)} événements récupérés.\n")
    return evenements


evenements  = telecharger_evenements()


# Générer les embeddings avec Mistral

def generer_embeddings(evenements):
    """Génère un vecteur (embedding) pour chaque description d'événement."""
 
    client = Mistral(api_key=MISTRAL_API_KEY)
 
    resultats = []
 
    for evenement in evenements:
        # On récupère le titre et la description
        titre      = evenement.get("title_fr") or evenement.get("title_en") or "Sans titre"
        description = evenement.get("description_fr") or evenement.get("description_en") or ""
 
        if not description:
            print(f"  Pas de description pour : {titre}")
            continue
         
        # Appel à l'API Mistral pour générer l'embedding
        reponse_mistral = client.embeddings.create(
            model="mistral-embed",
            inputs=[f"{titre}. {description}"],
        )
        vecteur = reponse_mistral.data[0].embedding
 
        # On stocke le résultat
        resultats.append({
            "titre":      titre,
            "ville":      evenement.get("location_city", ""),
            "date_debut": evenement.get("date_start", ""),
            "description": description[:200],   # Aperçu de la description
            "embedding":  vecteur,              # Vecteur de 1024 dimensions
        })
 
        print(f" {titre[:60]}")
        print(f"  Ville : {evenement.get('location_city', '?')}")
        print(f"  Vecteur : {len(vecteur)} dimensions — premiers chiffres : {vecteur[:4]}\n")
 
        time.sleep(0.3)  # pause pour ne pas surcharger l'API
 
    return resultats


resultats   = generer_embeddings(evenements)
 
print(f"\nTerminé ! {len(resultats)} embeddings générés.")


# Sauvegarder les résultats

def sauvegarder(resultats):
    """Sauvegarde les résultats dans deux fichiers :
    - CSV     : sans les vecteurs (trop volumineux)
    - Parquet : fichier complet avec les vecteurs"""
 
    # CSV — sans la colonne embedding
    df_csv = resultats.drop(columns=["embedding"])
    df_csv.to_csv("evenements_occitanie.csv", index=False, encoding="utf-8-sig")
    print(" Fichier CSV sauvegardé: evenements_occitanie.csv")
 
    # Parquet — avec les embeddings (format binaire optimisé pour Python)
    resultats.to_parquet("evenements_occitanie.parquet", index=False)
    print(" Fichier Parquet sauvegardé: evenements_occitanie.parquet")
    print('   df = pd.read_parquet("evenements_occitanie.parquet")')


