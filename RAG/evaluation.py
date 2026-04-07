

# Exact Match (recherche par mot-clé dans les métadonnées)
def exact_match(reponse, attendu):
    """Vérifie si la réponse correspond exactement à la valeur attendue."""
    return int(reponse.strip().lower() == attendu.strip().lower())

# score simplicité
def score_pertinence(docs):
    """Retourne le nombre de documents récupérés (proxy de pertinence)."""
    return len(docs)