# Récupérer les événements via l'API OpenDataSoft de Openagenda

import requests # appels requête HTTP pour récupérer les données de l'API
from datetime import datetime, timedelta, timezone
from collections import Counter


def telecharger_evenements():
    il_y_a_un_an = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    
    tous_les_evenements = []
    offset = 0

    while len(tous_les_evenements) < 200:
        params = {
            "where": f"location_city='Toulouse' AND firstdate_begin >= date'{il_y_a_un_an}'",
            "limit": 100,          # max autorisé par l'API opendatasoft
            "offset": offset,      # pagination
            "order_by": "firstdate_begin ASC",
        }
        reponse = requests.get(url, params=params)
        
        print(f"Statut HTTP : {reponse.status_code} (offset={offset})")
        if reponse.status_code != 200:
            print("Erreur API :", reponse.text[:500])
            break

        donnees = reponse.json()
        evenements = donnees.get("results", [])
        
        if not evenements:
            break  # plus de résultats

        tous_les_evenements.extend(evenements)
        offset += 100
        print(f"{len(tous_les_evenements)} événements récupérés au total...")

        if len(evenements) < 100:
            break  # dernière page atteinte

    # Filtrer les événements "test", "test xxx",etc
    tous_les_evenements = [
        e for e in tous_les_evenements
        if e.get("title_fr", "").lower().strip() not in ["test", "test charlotte", ""]
        and e.get("firstdate_begin", "") <= "2026-06-06T00:00:00+00:00"
    ]

    print(f"{len(tous_les_evenements)} événements récupérés après filtrage.\n")

    # Distribution des dates
    from collections import Counter
    mois = [e.get("firstdate_begin", "")[:7] for e in tous_les_evenements]
    for m, count in sorted(Counter(mois).items()):
        print(f"{m} : {count} événements")

    return tous_les_evenements

