# test embedding

import numpy as np
from RAG.embedding import get_embedding

def test_get_embedding(mocker):
    fake_response = mocker.Mock()
    fake_response.data = [mocker.Mock(embedding=[0.1]*1024)]

    mock_client = mocker.Mock()
    mock_client.embeddings.create.return_value = fake_response

    embedding = get_embedding(mock_client, "test")

    assert len(embedding) == 1024 # embedding de 1024 dimensions
