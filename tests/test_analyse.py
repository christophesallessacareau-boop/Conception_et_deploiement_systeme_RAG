# test de la fonction d'analyse exploratoire des événements
import pandas as pd
from main import analyser_evenements

def test_analyser_evenements_filtrage():
    # Données fake
    evenements = [
        {
            "title_fr": "Concert A",
            "location_city": "Toulouse",
            "location_region": "Occitanie",
            "description_fr": "Un concert sans intérêt",
            "firstdate_begin": "2024-01-01"
        },
        {
            "title_fr": "Event B",
            "location_city": "Montpellier",
            "location_region": "Occitanie",
            "description_fr": None,
            "firstdate_begin": "2024-02-01"
        },
        {
            "title_fr": "Festival C",
            "location_city": "Nîmes",
            "location_region": "Occitanie",
            "description_fr": "Festival incroyable",
            "firstdate_begin": "2024-03-01"
        }
    ]

    # Appel de la fonction
    df = analyser_evenements(evenements, verbose=False)

    # Vérifications
    assert isinstance(df, pd.DataFrame)

    # Il doit rester seulement 2 événements (ceux avec description)
    assert len(df) == 2

    # Vérifier qu'il n'y a plus de valeurs nulles
    assert df["description_fr"].isna().sum() == 0

    # Vérifier que les bons titres sont présents
    titres = df["title_fr"].tolist()
    assert "Concert A" in titres
    assert "Festival C" in titres
    assert "Event B" not in titres