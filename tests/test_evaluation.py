from RAG.evaluation import exact_match, score_pertinence


def test_exact_match_vrai():
    """Réponse identique à l'attendu : doit retourner 1."""
    assert exact_match("Toulouse", "Toulouse") == 1


def test_exact_match_casse():
    """Majuscules/minuscules différentes : doit quand même retourner 1."""
    assert exact_match("TOULOUSE", "toulouse") == 1


def test_exact_match_espaces():
    """Espaces en trop : doit quand même retourner 1."""
    assert exact_match("  Toulouse  ", "Toulouse") == 1


def test_exact_match_faux():
    """Réponse différente de l'attendu : doit retourner 0."""
    assert exact_match("Montpellier", "Toulouse") == 0


def test_score_pertinence_normal():
    """3 documents récupérés : score = 3."""
    docs = ["doc1", "doc2", "doc3"]
    assert score_pertinence(docs) == 3


def test_score_pertinence_vide():
    """Aucun document récupéré : score = 0."""
    assert score_pertinence([]) == 0