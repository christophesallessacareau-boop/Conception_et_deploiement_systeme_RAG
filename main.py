
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
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY) #
print("Clé chargée :", MISTRAL_API_KEY is not None)


# Pipeline principale du projet:
if __name__ == "__main__":

    #récupération des événements depuis l'API:
    evenements = telecharger_evenements()

    # analyse exploratoire:
    df = analyser_evenements(evenements)

    # embeddings:
    resultats = generer_embeddings(evenements,client)
    print(f"\nTerminé ! {len(resultats)} embeddings générés.")

    # Construction de l'index FAISS
    index, metadatas = construire_index_faiss(resultats)

    # Vérification que tout est bien indexé avec FAISS:
    print("Nombre de vecteurs FAISS :", index.ntotal)

    # Vérification que tout est bien indexé avec les métadonnées:
    print("Nombre de métadonnées :", len(metadatas))

    # Vérification que tout est bien indexé avec les résultats (chunks):
    print("Nombre de résultats (chunks) :", len(resultats))

    if index.ntotal == len(metadatas)== len(resultats):
        print(" Tous les événements sont bien indexés !")
    else:
        print(" Problème : mismatch index / metadata / résultats (chunks) !")

    # Vérification (titre, ville, date de debut, descripion) de la première métadonnée
    print("\n Exemple de métadonnée :")
    print(metadatas[0])



    # Exemple de recherche réel:
    query = "concert Toulouse"
    print(f"\nRecherche pour : '{query}'")
    query_embedding = get_embedding(client, query)
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
        temperature=0.3
    )
    rag = construire_chaine_rag(retriever, llm)


    # question test:
    question = "Quels événements à Toulouse ce week-end ?"
    reponse, docs = rag(question)
    print("\n Réponse :\n")
    print(reponse)


    # évaluation
    docs_scores = vectorstore.similarity_search_with_score(question, k=3)
    for doc, score in docs_scores:
        print(score, doc.metadata["titre"])

    print("Score pertinence :", score_pertinence(docs))


