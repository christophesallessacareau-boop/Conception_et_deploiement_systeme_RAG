# Reconstruit le pipeline de traitement des données en appelant l'endpoint d'administration.
# Appel automatique du mot de passe ADMIN de l'API FastAPI
# évite de taper curl ou d'utiliser Swagger

import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

BASE_URL = "http://127.0.0.1:8000"  # port fixe
ADMIN_KEY = os.getenv("ADMIN_KEY")

def rebuild_pipeline():
    print(f"Clé chargée : {repr(ADMIN_KEY)}")
    r = requests.post(
        f"{BASE_URL}/admin/rebuild",
        headers={"X-ADMIN-Key": ADMIN_KEY}
    )
    print(r.status_code, r.json())

if __name__ == "__main__":
    rebuild_pipeline()