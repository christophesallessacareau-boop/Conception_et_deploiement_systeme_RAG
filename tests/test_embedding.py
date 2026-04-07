# test embedding

import numpy as np
from unittest.mock import patch
from RAG.embedding import get_embedding, chunker_texte


def test_get_embedding():
    """Vérifie que get_embedding retourne bien un vecteur de 1024 dimensions."""

    with patch("RAG.embedding.Mistral") as MockMistral:
        # On configure le faux client Mistral
        fake_client = MockMistral.return_value
        fake_client.embeddings.create.return_value.data = [
            type("obj", (), {"embedding": [0.1] * 1024})()
        ]

        vecteur = get_embedding(fake_client, "test")

    assert vecteur is not None
    assert len(vecteur) == 1024


def test_chunker_texte_court():
    """Un texte court → 1 seul chunk."""
    chunks = chunker_texte("Bonjour, ceci est un texte court.")
    assert len(chunks) >= 1


def test_chunker_texte_long():
    """Un texte long → plusieurs chunks."""
    texte_long = "mot " * 300    # 300 mots répétés
    chunks = chunker_texte(texte_long)
    assert len(chunks) > 1