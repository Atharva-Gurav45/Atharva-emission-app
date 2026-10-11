import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# paths are relative to this file, so they work on your PC and on Render
Model_file = str(BASE_DIR / "project_app" / "emission_model.pkl")
JSON_file = str(BASE_DIR / "project_app" / "emission_model.json")

HOST = '0.0.0.0'
PORT = int(os.environ.get("PORT", 5006))   # Render gives the PORT; locally it stays 5006
