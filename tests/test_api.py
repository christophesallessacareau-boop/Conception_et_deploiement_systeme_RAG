# tests/test_api.py
# Tests fonctionnels de l'API FastAPI
# On utilise TestClient de FastAPI qui simule des appels HTTP sans vrai serveur

import os
import api
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Mocks AVANT l'import de api.py 
# api.py appelle build_pipeline() au démarrage : on le bloque
# pour éviter de vrais appels à Mistral pendant les tests

os.environ.setdefault("MISTRAL_API_KEY", "fake-mistral-key")
os.environ.setdefault("ADMIN_KEY",       "fake-admin-key")

# On bloque les imports lourds avant que api.py ne les charge
with patch("RAG.retrieval.telecharger_evenements", return_value=[]):
    with patch("api.build_pipeline"):          # bloque le démarrage du pipeline
        from api import app

client_test = TestClient(app)
ADMIN_KEY   = os.environ["ADMIN_KEY"]


# TESTS HEALTH CHECK

def test_health_check():
    """ endpointGET / doit retourner 200 et un message de statut."""
    response = client_test.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message"  in data
    assert "pipeline" in data


# TESTS /ask

def test_ask_question_vide():
    """endpointPOST /ask: une question vide doit retourner 400."""
    response = client_test.post("/ask", json={"question": ""})
    assert response.status_code == 400
    assert "vide" in response.json()["detail"].lower()

def test_ask_question_espaces():
    """endpointPOST /ask:
    Une question avec seulement des espaces doit retourner 400."""
    response = client_test.post("/ask", json={"question": "   "})
    assert response.status_code == 400

def test_ask_pipeline_non_initialise():
    """endpointPOST /ask:
    Si le pipeline n'est pas prêt, /ask doit retourner 503."""
    # On s'assure que vectorstore est None (pipeline non initialisé)
    api.vectorstore = None

    response = client_test.post("/ask", json={"question": "Quels concerts à Toulouse ?"})
    assert response.status_code == 503
    assert "non initialisé" in response.json()["detail"].lower()

def test_ask_pipeline_pret():
    """endpoint POST /ask:
    Avec un pipeline simulé, /ask doit retourner 200 avec answer et sources."""
    
    # On simule un vectorstore et une rag_chain fonctionnels
    api.vectorstore = MagicMock()

    fake_doc = MagicMock()
    fake_doc.metadata = {"titre": "Jazz Fest", "ville": "Toulouse", "date_debut": "2025-06-01"}

    # On remplace creer_retriever et construire_chaine_rag dans api.py
    with patch("api.creer_retriever") as mock_retriever, \
         patch("api.construire_chaine_rag") as mock_rag:

        # La chaîne RAG retourne une réponse simulée
        mock_rag.return_value = lambda q: ("Voici des événements à Toulouse.", [fake_doc])

        response = client_test.post("/ask", json={"question": "Quels concerts à Toulouse ?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer"   in data
    assert "sources"  in data
    assert "question" in data
    assert data["answer"] == "Voici des événements à Toulouse."
    assert data["sources"][0]["titre"] == "Jazz Fest"


# TESTS /admin/rebuild

def test_rebuild_sans_cle():
    """endpointPOST /admin/rebuild:
    Sans clé admin, /admin/rebuild doit retourner 403."""
    response = client_test.post("/admin/rebuild")
    assert response.status_code in [401, 403]

def test_rebuild_mauvaise_cle():
    """endpointPOST /admin/rebuild:
    Avec une mauvaise clé, /admin/rebuild doit retourner 401."""
    response = client_test.post(
        "/admin/rebuild",
        headers={"X-ADMIN-Key": "mauvaise-cle"}
    )
    assert response.status_code == 401

def test_rebuild_bonne_cle():
    """endpointPOST /admin/rebuild:
    Avec la bonne clé admin, /admin/rebuild doit retourner 200."""
    api.last_rebuild = 0   # reset du rate limit

    with patch("api.build_pipeline") as mock_build:
        mock_build.return_value = None   # on simule un rebuild réussi
        response = client_test.post(
            "/admin/rebuild",
            headers={"X-ADMIN-Key": ADMIN_KEY}
        )

    assert response.status_code == 200
    assert "reconstruit" in response.json()["status"].lower()

def test_rebuild_rate_limit():
    """endpointPOST /admin/rebuild:
    Deux rebuilds consécutifs rapides doivent déclencher un 429."""
    api.last_rebuild = 0   # reset

    with patch("api.build_pipeline"):
        # Premier rebuild — OK
        client_test.post("/admin/rebuild", headers={"X-ADMIN-Key": ADMIN_KEY})

        # Deuxième rebuild immédiat — doit être bloqué
        response = client_test.post("/admin/rebuild", headers={"X-ADMIN-Key": ADMIN_KEY})

    assert response.status_code == 429