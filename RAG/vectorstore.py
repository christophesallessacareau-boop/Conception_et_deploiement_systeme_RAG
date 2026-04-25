# LangChain attend un vectorstore avec une méthode de recherche
# vectorstore compatible LongChain
# on crée une classe pour notre index FAISS et les métadonnées pour l'utiliser dans LangChain:

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from RAG.embedding import get_embedding


def construire_vectorstore_langchain(resultats, client):
    """Construit un vectorstore LangChain/FAISS à partir des chunks déjà embeddés."""

    documents        = []
    embeddings_matrix = np.array(
        [r["embedding"] for r in resultats]
    ).astype("float32")

    for r in resultats:
        documents.append(
            Document(
                page_content=r["description"],
                metadata={
                    "titre": r["titre"],
                    "ville": r["ville"],
                    "date_debut":  r["date_debut"],
                }
            )
        )
    # Embeddings est la classe de base LongChain
    # FAISS sait alors appeler la classe avec 2 méthodes
    class FakeEmbeddings(Embeddings): 
        
        def embed_documents(self, texts): # liste de textes à indexer
            return embeddings_matrix.tolist() # vecteurs déjà calculés, on évite de les recalculer ici

        def embed_query(self, text): # question posée par l'utilisateur
            return get_embedding(client, text)  # vrai appel Mistral et retour d'un vecteur

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(
            [doc.page_content for doc in documents], # textes
            embeddings_matrix.tolist() #vecteurs déjà calculés
        )),
        embedding=FakeEmbeddings(),
        metadatas=[doc.metadata for doc in documents],
    )

    print(f" Vectorstore prêt : {len(documents)} documents")
    return vectorstore


def creer_retriever(vectorstore, ville=None, k=5):
    """Crée un retriever avec filtre optionnel sur la ville.
    Le retriever retourne les documents les plus proches vectoriellement.
    On indique la ville si besoin.
    k : vecteurs les plus proches du texte de la question.
    k : nombre de résultats à retourner."""
    
    if ville:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {"ville": ville}  # filtre par métadata "ville"
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
    return retriever