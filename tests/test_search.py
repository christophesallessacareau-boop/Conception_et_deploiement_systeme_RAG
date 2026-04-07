# Test recherche

import numpy as np
from RAG.search import rechercher, construire_index_faiss

def test_rechercher():
    resultats = [
        {"embedding": [0.1]*1024, "titre": "Concert", "ville": "Toulouse", "date_debut": "2024", "description": "musique"}
    ]

    index, metadatas = construire_index_faiss(resultats)

    query_embedding = [0.1]*1024

    results = rechercher(index, metadatas, query_embedding)

    assert len(results) > 0