# Sauvegarder les résultats

import os
import pandas as pd

def sauvegarder(resultats):
    """Sauvegarde les résultats dans deux fichiers :
    - CSV     : sans les vecteurs (trop volumineux)
    - Parquet : fichier complet avec les vecteurs"""
 
    # CSV — sans la colonne embedding
    df = pd.DataFrame(resultats)
    df_csv = df.drop(columns=["embedding"])
    df_csv.to_csv("evenements_occitanie.csv", index=False, encoding="utf-8-sig")
    print(" Fichier CSV sauvegardé: evenements_occitanie.csv")
 
    # Parquet — avec les embeddings (format binaire optimisé pour Python)
    df.to_parquet("evenements_occitanie.parquet", index=False)
    print(" Fichier Parquet sauvegardé dans /sauvegarde/")
    print('   df = pd.read_parquet("evenements_occitanie.parquet")')