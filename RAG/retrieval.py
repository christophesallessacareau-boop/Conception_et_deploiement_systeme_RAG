# Récupérer les événements via l'API OpenDataSoft de Openagenda

import requests # appels requête HTTP pour récupérer les données de l'API
from datetime import datetime, timedelta, timezone


def telecharger_evenements():
    """Récupère les événements d'Occitanie via l'API OpenDataSoft de Openagenda."""
 
    # Date d'il y a 1 an (format attendu par l'API)
    il_y_a_un_an = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
 
    # Paramètres de la requête (localisation et date 1 an historique et à venir)
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    params = {
        "where": f"location_region='Occitanie' AND firstdate_begin >= date'{il_y_a_un_an}'",
        "limit": 25,           # pour 25 événements
        "order_by": "firstdate_begin DESC", # trier par date décroissante
    }
 
    print("Téléchargement des événements en cours")
    reponse = requests.get(url, params=params) # envoie de la requête au serveur

    # afficher le statut pour détecter les erreurs de l'API:
    print("Statut HTTP :", reponse.status_code)
    if reponse.status_code != 200:
        print(" Erreur API :", reponse.text[:500])
        return [] # liste vide si erreur
    
    donnees = reponse.json()
 
    evenements = donnees.get("results", []) # on retourne une liste vide si pas de résultats
    print(f"{len(evenements)} événements récupérés.\n")
    return evenements

