# Générer les embeddings avec Mistral
## Ce module s'occupe de transformer les descriptions d'événements en vecteurs numériques (embeddings).
### pour chaque événement, découpe en chunks et génère les vecteurs
#### Retourne une liste avec titre, ville, date de début,description, embedding

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral

## Chunking: descriptions longues en morceaux plus petits pour éviter les dépassements de limite de tokens

def chunker_texte(texte):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 500 caract max par chunk
        chunk_overlap=50 # chevauchement entre chunks N et N+1 avec les 50 derniers caractères
    )
    return splitter.split_text(texte)


def get_embedding(client, texte, retries=5): # limitation de débit fréquent avec Mistral AI=> retry auto
    """Appelle l'API Mistral avec retry automatique en cas d'erreur (ex: 429: too many requests)."""
    for i in range(retries):
        try:
            response = client.embeddings.create(
                model="mistral-embed",
                inputs=[texte]
            )
            return response.data[0].embedding

        except Exception as e:
            print(f" Erreur API (tentative {i+1}) :", e)
            time.sleep(2 * (i + 1))  # temporisation de plus en plus importante entre les tentatives

    print(" Échec après plusieurs tentatives.")
    return None # en cas d'échec après tous les retries, on retourne None pour éviter de planter le programme


def generer_embeddings(evenements, client):
    """Génère un vecteur (embedding) pour chaque description d'événement."""
 
    
    resultats = []
 
    for evenement in evenements:
        # On récupère le titre et la description
        titre      = evenement.get("title_fr") or evenement.get("title_en") or "Sans titre"
        description = evenement.get("description_fr") or evenement.get("description_en") or ""
 
        if not description:
            print(f"  Pas de description pour : {titre}")
            continue # on passe à l'événement suivant si pas de description

        chunks=chunker_texte(description) # on découpe la description en morceaux
         
        for chunk in chunks:
        # appel via fonction avec retry
            texte = f"{titre}. {chunk}" # on associe le chunk avec le titre pour aider à vectoriser

            vecteur = get_embedding(client, texte)
            if vecteur is None:
                print(f" Embedding échoué pour : {titre}")
                continue # on continue même si une requête échoue
         
        # On stocke le résultat
            resultats.append({
                "titre":      titre,
                "ville":      evenement.get("location_city", ""),
                "date_debut": evenement.get("firstdate_begin", ""), 
                "description": chunk, # on indexe des morceaux et non pas la description entière
                "embedding":  vecteur,              # Vecteur de 1024 dimensions
            })
 
            print(f" {titre[:60]}")
            print(f"  Ville : {evenement.get('location_city', '?')}")
            print(f"  Vecteur : {len(vecteur)} dimensions — premiers chiffres : {vecteur[:4]}\n")
 
            time.sleep(1)  # pause pour limiter les appels à l'API
 
    return resultats


