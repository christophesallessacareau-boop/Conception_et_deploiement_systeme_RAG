# Analyse Exploratoire des données

import pandas as pd


def analyser_evenements(evenements, verbose=True):
    """
    Transforme les événements en DataFrame:
    - Combien d'événements ont une description ?
    - Quelles colonnes sont souvent vides ?
    - Aperçu des données
    """

    # Conversion en DataFrame pour analyse
    df = pd.DataFrame(evenements)
    
    # Aperçu des colonnes utiles
    colonnes_utiles = ["title_fr", "location_city", "location_region", "location_departement", "firstdate_begin", 
                       "firstdate_end", "lastdate_begin", "lastdate_end", "description_fr", "location_name"]
    colonnes_presentes = [c for c in colonnes_utiles if c in df.columns]
    print(df[colonnes_presentes].head(5).to_string())

    # données manquantes
    taux_manquants = (df[colonnes_presentes].isnull().mean() * 100).round(1)
    print(taux_manquants.to_string())

    # zoom sur les descriptions (colonne la plus importante)
    if "description_fr" in df.columns:
        nb_avec = df["description_fr"].notna().sum()
        nb_sans = df["description_fr"].isna().sum()
        print(f"   - Avec description : {nb_avec}")
        print(f"   - Sans description : {nb_sans}")

    # On ne garde que les événements avec une description
    nb_evenement = len(df)
    if "description_fr" in df.columns:
        df = df[df["description_fr"].notna()].reset_index(drop=True)
    if verbose:
        print(f"\n Après filtrage (avec description uniquement) : {len(df)} / {nb_evenement} événements\n")

    return df

