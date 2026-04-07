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
                    "date":  r["date_debut"],
                }
            )
        )

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embeddings_matrix.tolist()   # vecteurs déjà calculés

        def embed_query(self, text):
            return get_embedding(client, text)  # vrai appel Mistral

    vectorstore = FAISS.from_texts(
    texts=[doc.page_content for doc in documents],
    embedding=FakeEmbeddings(),
    metadatas=[doc.metadata for doc in documents],
)

    print(f" Vectorstore prêt : {len(documents)} documents")
    return vectorstore


def creer_retriever(vectorstore):
    """Crée un retriever LangChain qui retourne les 3 documents les plus proches."""
    return vectorstore.as_retriever(search_kwargs={"k": 3})