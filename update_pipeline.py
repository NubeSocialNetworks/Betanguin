import os
import pandas as pd
from src.elo import compute_elo
from src.features import add_features

print("1. Descargando datos...")
os.system("python src/download_data.py")

print("2. Procesando datos...")
os.system("python src/process_data.py")

df = pd.read_csv("data/processed/matches.csv")

print("3. Calculando ELO...")
df = compute_elo(df)

print("4. Añadiendo features...")
df = add_features(df)
df.to_csv("data/processed/final_dataset.csv", index=False)

print("5. Entrenando modelos...")
os.system("python src/train_models.py")

print("✔ Pipeline completado")
