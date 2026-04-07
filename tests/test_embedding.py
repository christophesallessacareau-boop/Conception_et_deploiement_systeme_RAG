# test embedding

from unittest.mock import MagicMock

import numpy as np
from RAG.embedding import get_embedding
from unittest.mock import MagicMock

def test_get_embedding(mocker):
    fake_response = mocker.Mock()
    fake_response.data = [mocker.Mock(embedding=[0.1]*1024)]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
    MagicMock(embedding=[0.1]*1024)
]

    embedding = get_embedding(mock_client, "test")

    assert len(embedding) == 1024 # embedding de 1024 dimensions
