# vérification des fichiers créés et du contenuer contenu

import os
import pandas as pd

from RAG.save import sauvegarder

def test_sauvegarder(tmp_path): # pytest fournit un dossier temporaire pour ce test
    # Données fake
    resultats = [
        {
            "titre": "Test",
            "ville": "Toulouse",
            "date_debut": "2024",
            "description": "desc",
            "embedding": [0.1]*10
        }
    ]

    # Se placer dans un dossier temporaire
    os.chdir(tmp_path)

    # Appel fonction
    sauvegarder(resultats)

    # Vérifier fichiers créés
    assert os.path.exists("evenements_occitanie.csv")
    assert os.path.exists("evenements_occitanie.parquet")

    # Vérifier contenu CSV
    df_csv = pd.read_csv("evenements_occitanie.csv")
    assert len(df_csv) == 1
    assert "embedding" not in df_csv.columns

    # Vérifier contenu Parquet
    df_parquet = pd.read_parquet("evenements_occitanie.parquet")
    assert len(df_parquet) == 1
    assert "embedding" in df_parquet.columns