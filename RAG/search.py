# Recherche sémantique

import numpy as np



def rechercher(index, metadatas, query_embedding, k=3):
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(metadatas[i])

    return results