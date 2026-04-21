# Appels des fonctions dans l'ordre du projet RAG, affichage des résultats er gestions des variables globales

import sys
import os # acces aux variables d'environnement
import pandas as pd
import numpy as np
import streamlit as st # interface UI

from dotenv import load_dotenv
import time
import requests
from datetime import datetime, timedelta, timezone

import faiss
from mistralai import Mistral
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # classe de base des embeddings LangChain
from langchain_community.vectorstores import FAISS

# import des fonctions du code du projet:
from RAG.retrieval import telecharger_evenements
from RAG.analyse import analyser_evenements
from RAG.embedding import chunker_texte, get_embedding, generer_embeddings
from RAG.faiss import construire_index_faiss
from RAG.search import rechercher
from RAG.save import sauvegarder
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever
from RAG.rag         import construire_chaine_rag
from RAG.evaluation  import exact_match, score_pertinence

print ("imports réalisés")


# Récupération de la clé API Mistral depuis le fichier .env:
load_dotenv() # lit lefichier .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") # récupération de la clé dans .env
client = Mistral(api_key=MISTRAL_API_KEY) # objet Mistral pour faire les appels à l'API
print("Clé chargée :", MISTRAL_API_KEY is not None) # True si clé a été retrouvée


# Pipeline principale du projet:
## Exécute ce bloc UNIQUEMENT si on lance ce fichier directement.
if __name__ == "__main__":

    #récupération des événements depuis l'API (appel à la fonction du module retrieval.py):
    evenements = telecharger_evenements()

    # analyse exploratoire (appel de analyse.py):
    df = analyser_evenements(evenements)

    # embeddings (pour chaque événement, découpe en chunks et génère les vecteurs):
    resultats = generer_embeddings(evenements,client)
    print(f"\nTerminé ! {len(resultats)} embeddings générés.")

    # Construction de l'index de recherche FAISS (appel de faiss.py):
    ## retourne un index FAISS et les métadonnées associées à chaque vecteur (titre, ville, date, description)
    index, metadatas = construire_index_faiss(resultats)

    # Vérification que tout est bien indexé avec FAISS:
    print("Nombre de vecteurs FAISS :", index.ntotal)

    # Vérification que tout est bien indexé avec les métadonnées:
    print("Nombre de métadonnées :", len(metadatas))

    # Vérification que tout est bien indexé avec les résultats (chunks):
    print("Nombre de résultats (chunks) :", len(resultats))

    # FAISS et les métadonnées sont deux structures séparées alignées par position. 
    # Si leurs tailles diffèrent, la recherche retournerait de mauvaises métadonnées pour un vecteur donné
    if index.ntotal == len(metadatas)== len(resultats):
        print(" Tous les événements sont bien indexés !")
    else:
        print(" Problème : mismatch index / metadata / résultats (chunks) !")

    # Vérification (titre, ville, date de debut, descripion) de la première métadonnée
    print("\n Exemple de métadonnée :")
    print(metadatas[0])



    # Exemple de recherche FAISS brute (sans LLM):
    query = "événements dans la ville de Toulouse en 2026"
    print(f"\nRecherche brute FAISS sans LLM pour : '{query}'")
    # on transforme la question en vecteur:
    query_embedding = get_embedding(client, query)
    # FAISS trouve les vecteurs les plus proches et retourne les métadonnées brutes associées:
    results = rechercher(index, metadatas, query_embedding)

    for r in results:
        print(r["titre"], "-", r["ville"])


    # Vectorstore LangChain:
    vectorstore = construire_vectorstore_langchain(resultats,client)
    retriever = creer_retriever(vectorstore)


    # Initialiser Mistral avec LangChain, chaîne RAG
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-small-latest",
        temperature=0
    )
    rag = construire_chaine_rag(retriever, llm)


    # question test augmentée (avec LLM Mistral):
    question = "Quels événements à Toulouse en 2026 ?"
    reponse, docs = rag(question)
    print("\n Réponse RAG complète:\n")
    print(reponse)


    # évaluation
    docs_scores = vectorstore.similarity_search_with_score(question, k=5)
    for doc, score in docs_scores:
        print(score, doc.metadata["titre"], doc.metadata["ville"])

    print("Score pertinence :", score_pertinence(docs))


