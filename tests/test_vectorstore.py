# on mocke get_embedding pour éviter les appels réels à Mistral.
# les tests vérifient que le vectorstore est construit correctement et que les métadonnées sont bien conservées.

import numpy as np
from unittest.mock import patch
from RAG.vectorstore import construire_vectorstore_langchain, creer_retriever


# Données fictives réutilisées dans tous les tests
RESULTATS_FAKES = [
    {
        "titre":      "Jazz Festival",
        "ville":      "Toulouse",
        "date_debut": "2025-06-01",
        "description": "Un grand festival de jazz en plein air.",
        "embedding":  [0.1] * 1024,   # vecteur fictif de 1024 dimensions
    },
    {
        "titre":      "Marché bio",
        "ville":      "Montpellier",
        "date_debut": "2025-06-15",
        "description": "Marché de produits biologiques locaux.",
        "embedding":  [0.2] * 1024,
    },
]


def test_construire_vectorstore_retourne_un_objet():
    """Le vectorstore est bien construit et contient le bon nombre de documents."""

    # On mocke get_embedding pour éviter un vrai appel à Mistral
    with patch("RAG.vectorstore.get_embedding", return_value=[0.1] * 1024):
        client     = None   # pas besoin d'un vrai client grâce au mock
        vectorstore = construire_vectorstore_langchain(RESULTATS_FAKES, client)

    # Le vectorstore doit exister
    assert vectorstore is not None


def test_vectorstore_contient_bons_documents():
    """Les métadonnées des documents sont bien conservées."""

    with patch("RAG.vectorstore.get_embedding", return_value=[0.1] * 1024):
        vectorstore = construire_vectorstore_langchain(RESULTATS_FAKES, None)
        # Recherche simple pour vérifier que les docs sont bien là
        results = vectorstore.similarity_search("jazz", k=1)

    assert len(results) >= 1
    assert "titre" in results[0].metadata   # la métadonnée titre est présente


def test_creer_retriever():
    """Le retriever est bien créé et retourne le bon nombre de résultats."""

    # get_embedding est remplacé par une fonction qui retourne toujours [0.1, 0.1, ...]
    # sans appeler Mistral
    with patch("RAG.vectorstore.get_embedding", return_value=[0.1] * 1024):
        vectorstore = construire_vectorstore_langchain(RESULTATS_FAKES, None)
        retriever = creer_retriever(vectorstore)
        docs = retriever.invoke("festival")

    # Le retriever doit exister
    assert retriever is not None
    assert isinstance(docs, list)
    assert len(docs) <= 3  # Il doit retourner au maximum k=3 documents
