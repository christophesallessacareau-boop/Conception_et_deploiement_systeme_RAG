# Générer les embeddings avec Mistral

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral

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


def generer_embeddings(evenements, client):
    """Génère un vecteur (embedding) pour chaque description d'événement."""
 
    
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
                "date_debut": evenement.get("firstdate_begin", ""), # date_start n'existe pas
                "description": chunk, # plutôt que description[:200] par ex pour garder le chunk entier
                "embedding":  vecteur,              # Vecteur de 1024 dimensions
            })
 
            print(f" {titre[:60]}")
            print(f"  Ville : {evenement.get('location_city', '?')}")
            print(f"  Vecteur : {len(vecteur)} dimensions — premiers chiffres : {vecteur[:4]}\n")
 
            time.sleep(1)  # pause pour limiter les appels à l'API
 
    return resultats


