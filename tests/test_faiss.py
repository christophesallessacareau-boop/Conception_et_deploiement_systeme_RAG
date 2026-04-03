# Test index FAISS:
from main import construire_index_faiss
import numpy as np

def test_construire_index_faiss():
    # Test d'un index FAISS avec un résultat simulé
    resultats = [
        {
            "titre": "Événement 1",
            "ville": "Toulouse",
            "date_debut": "2023-01-01",
            "description": "Description 1",
            "embedding": [0.1] * 1024  # Simulate an embedding
        }
    ]
    index, metadatas = construire_index_faiss(resultats)
    assert index.ntotal == 1
    assert len(metadatas) == 1

