# convertir les données en index FAISS
# FAISS: moteur de recherche de notre RAG:
## pour retrouver rapidement les vecteurs les plus proches
### retourne un index FAISS et les métadonnées associées à chaque vecteur (titre, ville, date, description)

import faiss
import numpy as np



def construire_index_faiss(resultats):
    """Construit un index FAISS + stocke les métadonnées séparément."""
    
    # Vecteurs transformés en matrice NumPy (1 vecteur = 1 ligne)
    ## FAISS n'accepte que ce format précis et OBLIGATOIRE (float32)
    vectors = np.array([r["embedding"] for r in resultats]).astype("float32")
    
    # Dimension des vecteurs (1024 pour embeddings Mistral)
    dim = vectors.shape[1]

    # Index FAISS
    ## L2 = distance euclidienne entre vecteur requête et vecteurs stockés
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Métadonnées
    ## FAISS ne gère pas les métadonnées, on les stocke à part dans une liste de dictionnaires alignée à l'index
    metadatas = [
        {
            "titre": r["titre"],
            "ville": r["ville"],
            "date": r["date_debut"],
            "description": r["description"]
        }
        for r in resultats
    ]

    print(f" Index FAISS construit : {index.ntotal} vecteurs")
    
    # On retourne l'index et les métadonnées
    ## FAISS indique la position du vecteur le plus proche
    ### on utilise cette position pour récupérer les métadonnées correspondantes
    return index, metadatas