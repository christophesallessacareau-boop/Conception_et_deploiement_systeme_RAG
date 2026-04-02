
import sys
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import time
import requests
from datetime import datetime, timedelta, timezone

import faiss
from mistralai import Mistral
print("OK mistralai import") # pour verification de l'import de mistralai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print ("imports OK")



# Récupération de la clé API Mistral depuis le fichier .env:

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)
print("Clé chargée :", MISTRAL_API_KEY is not None)



# convertir les données en index FAISS

def construire_index_faiss(resultats):
    """Construit un index FAISS + stocke les métadonnées séparément."""
    
    # Vecteurs
    vectors = np.array([r["embedding"] for r in resultats]).astype("float32")
    
    dim = vectors.shape[1]

    # Index FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Métadonnées
    metadatas = [
        {
            "titre": r["titre"],
            "ville": r["ville"],
            "date": r["date_debut"],
            "description": r["description"]
        }
        for r in resultats
    ]

    print(f" Index FAISS construit : {index.ntotal} vecteurs")
    
    return index, metadatas



# Récupérer les événements via l'API OpenDataSoft

def telecharger_evenements():
    """Récupère les événements d'Occitanie via l'API OpenDataSoft de Openagenda."""
 
    # Date d'il y a 1 an (format attendu par l'API)
    il_y_a_un_an = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
 
    # Paramètres de la requête
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    params = {
        "where": f"location_region='Occitanie' AND firstdate_end >= date'{il_y_a_un_an}'",
        "limit": 50,           # pour 50 événements
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


# Analyse Exploratoire des données

def analyser_evenements(evenements):
    """
    Transforme les événements en DataFrame:
    - Combien d'événements ont une description ?
    - Quelles colonnes sont souvent vides ?
    - Aperçu des données
    """

    # Conversion en DataFrame pour analyse
    df = pd.DataFrame(evenements)
    
    # Aperçu des colonnes utiles
    colonnes_utiles = ["title_fr", "location_city", "location_region", "location_departement", "firstdate_begin", 
                       "firstdate_end", "lastdate_begin", "lastdate_end", "description_fr", "location_name"]
    colonnes_presentes = [c for c in colonnes_utiles if c in df.columns]
    print(df[colonnes_presentes].head(5).to_string())

    # données manquantes
    taux_manquants = (df[colonnes_presentes].isnull().mean() * 100).round(1)
    print(taux_manquants.to_string())

    # zoom sur les descriptions (colonne la plus importante)
    if "description_fr" in df.columns:
        nb_avec = df["description_fr"].notna().sum()
        nb_sans = df["description_fr"].isna().sum()
        print(f"   - Avec description : {nb_avec}")
        print(f"   - Sans description : {nb_sans}")

    # On ne garde que les événements avec une description
    nb_evenement = len(df)
    df = df[df["description_fr"].notna()].reset_index(drop=True)
    print(f"\n Après filtrage (avec description uniquement) : {len(df)} / {nb_evenement} événements\n")

    return df

df = analyser_evenements(evenements)



# Générer les embeddings avec Mistral

## Chunking:

def chunker_texte(texte):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(texte)


def get_embedding(client, texte, retries=5): # limitation de débit fréquent avec Mistral AI=> retry auto
    """Appelle l'API Mistral avec retry automatique en cas d'erreur (ex: 429)."""
    for i in range(retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[texte]
            )
            return response.data[0].embedding

        except Exception as e:
            print(f" Erreur API (tentative {i+1}) :", e)
            time.sleep(2 * (i + 1))  # backoff (temporisation)

    print(" Échec après plusieurs tentatives.")
    return None


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

        chunks=chunker_texte(description)
         
        for chunk in chunks:
        # appel via fonction avec retry
            texte = f"{titre}. {chunk}"

            vecteur = get_embedding(client, texte)
            if vecteur is None:
                print(f" Embedding échoué pour : {titre}")
                continue # on continue même si une requête échoue
         
        # On stocke le résultat
            resultats.append({
                "titre":      titre,
                "ville":      evenement.get("location_city", ""),
                "date_debut": evenement.get("date_start", ""),
                "description": chunk, # plutôt que description[:200] pour garder le chunk entier
                "embedding":  vecteur,              # Vecteur de 1024 dimensions
            })
 
            print(f" {titre[:60]}")
            print(f"  Ville : {evenement.get('location_city', '?')}")
            print(f"  Vecteur : {len(vecteur)} dimensions — premiers chiffres : {vecteur[:4]}\n")
 
            time.sleep(1)  # pause pour limiter les appels à l'API
 
    return resultats


resultats   = generer_embeddings(evenements)
print(f"\nTerminé ! {len(resultats)} embeddings générés.")


# Construction de l'index FAISS
index, metadatas = construire_index_faiss(resultats)

# Vérification que tout est bien indexé:
print("Nombre de métadonnées :", len(metadatas))

# Vérification que tout est bien indexé avec FAISS:
print("Nombre de vecteurs :", index.ntotal)

if index.ntotal == len(metadatas):
    print(" Tout est bien aligné !")
else:
    print(" Problème : mismatch index / metadata")



# Recherche sémantique

def rechercher(index, metadatas, query_embedding, k=3):
    import numpy as np

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(metadatas[i])

    return results

# Exemple de recherche:
query = "festival musique Toulouse"
print(f"\nRecherche pour : '{query}'")
query_embedding = get_embedding(client, query)

results = rechercher(index, metadatas, query_embedding)

for r in results:
    print(r["titre"], "-", r["ville"])



# Sauvegarder les résultats

def sauvegarder(resultats):
    """Sauvegarde les résultats dans deux fichiers :
    - CSV     : sans les vecteurs (trop volumineux)
    - Parquet : fichier complet avec les vecteurs"""
 
    # CSV — sans la colonne embedding
    df = pd.DataFrame(resultats)
    df_csv = df.drop(columns=["embedding"])
    df_csv.to_csv("evenements_occitanie.csv", index=False, encoding="utf-8-sig")
    print(" Fichier CSV sauvegardé: evenements_occitanie.csv")
 
    # Parquet — avec les embeddings (format binaire optimisé pour Python)
    df.to_parquet("evenements_occitanie.parquet", index=False)
    print(" Fichier Parquet sauvegardé: evenements_occitanie.parquet")
    print('   df = pd.read_parquet("evenements_occitanie.parquet")')


