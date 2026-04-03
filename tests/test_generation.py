# test generation
import numpy as np
from main import rechercher

def test_generation():
    response = "Voici des événements à Toulouse"

    assert isinstance(response, str)
    assert len(response) > 0