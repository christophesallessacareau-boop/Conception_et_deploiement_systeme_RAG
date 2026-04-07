# test récupération des événements:

from RAG.retrieval import telecharger_evenements

def test_telecharger_evenements():
    evenements = telecharger_evenements()
    assert isinstance(evenements, list)
    assert len(evenements) > 0