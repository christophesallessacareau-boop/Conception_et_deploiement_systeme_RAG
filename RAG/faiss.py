# convertir les données en index FAISS
import faiss
import numpy as np



def construire_index_faiss(resultats):
    """Construit un index FAISS + stocke les métadonnées séparément."""
    
    # Vecteurs
    vectors = np.array([r["embedding"] for r in resultats]).astype("float32")
    
    dim = vectors.shape[1]

    # Index FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Métadonnées
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
    
    return index, metadatas