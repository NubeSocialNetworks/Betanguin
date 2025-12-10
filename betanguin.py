import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from github import Github, Auth
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # PARA GITHUB ACTIONS
GITHUB_USER = "NubeSocialNetworks"
GITHUB_REPO = "Betanguin"

LEAGUES = {
    "LaLiga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "LaLiga2": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    "SerieA": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Ligue1": "https://www.football-data.co.uk/mmz4281/2526/FR1.csv",
    "Ligue2": "https://www.football-data.co.uk/mmz4281/2526/FR2.csv",
    "Bundesliga1": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Bundesliga2": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
    "PremierLeague": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Championship": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
}

# -------------------------------
# CREAR CARPETAS
# -------------------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# -------------------------------
# DESCARGA DE DATOS
# -------------------------------
for league, url in LEAGUES.items():
    try:
        r = requests.get(url)
        if r.status_code == 200:
            path = f"data/raw/{league}.csv"
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Descargado {league}")
        else:
            print(f"⚠ Error descargando {league}")
    except Exception as e:
        print(f"⚠ Error descargando {league}: {e}")

# -------------------------------
# PROCESAMIENTO Y FEATURE ENGINEERING
# -------------------------------
all_data = []
for file in os.listdir("data/raw"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/raw/{file}")
        cols_needed = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HY","AY","HR","AR"]
        available_cols = [c for c in cols_needed if c in df.columns]
        if len(available_cols) > 0:
            df = df[available_cols]
            all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Convertir Date
if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# BTTS
data["BTTS"] = ((data["FTHG"] > 0) & (data["FTAG"] > 0)).astype(int)

# Promedios
data["HomeGoalsAvg"] = data.groupby("HomeTeam")["FTHG"].transform("mean")
data["AwayGoalsAvg"] = data.groupby("AwayTeam")["FTAG"].transform("mean")
data["GoalDiff"] = data["HomeGoalsAvg"] - data["AwayGoalsAvg"]

# Tarjetas promedio
for t in ["HY", "AY", "HR", "AR"]:
    if t in data.columns:
        data[f"{t}_avg"] = (
            data.groupby("HomeTeam")[t].transform("mean")
            if t.startswith("H")
            else data.groupby("AwayTeam")[t].transform("mean")
        )

# Guardar dataset
data.to_csv("data/processed/full_dataset_enriched.csv", index=False)
print("✅ Dataset procesado y enriquecido")

# -------------------------------
# ENTRENAMIENTO MODELOS BASE
# -------------------------------
features = ["HomeGoalsAvg", "AwayGoalsAvg", "GoalDiff"] + [
    f"{t}_avg" for t in ["HY", "AY", "HR", "AR"] if f"{t}_avg" in data.columns
]

X = data[features]
y_result = data["FTR"].map({"H": 0, "D": 1, "A": 2})
y_btts = data["BTTS"]

model_result = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model_result.fit(X, y_result)

model_btts = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model_btts.fit(X, y_btts)

print("✅ Modelos XGB entrenados")

# -------------------------------
# POISSON
# -------------------------------
def poisson_match_probs(home_avg, away_avg, max_goals=10):
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob_matrix[i, j] = poisson.pmf(i, home_avg) * poisson.pmf(j, away_avg)
    return (
        np.sum(np.tril(prob_matrix, -1)),
        np.sum(np.diag(prob_matrix)),
        np.sum(np.triu(prob_matrix, 1)),
    )

def poisson_btts(home_avg, away_avg):
    return (1 - np.exp(-home_avg)) * (1 - np.exp(-away_avg))

# -------------------------------
# META-MODELO
# -------------------------------
def build_meta_dataset(data):
    rows = []
    for idx, row in data.iterrows():
        home_avg = row["HomeGoalsAvg"]
        away_avg = row["AwayGoalsAvg"]

        # XGB
        xgb_input = np.array([[home_avg, away_avg, row["GoalDiff"]]])
        pred_xgb = model_result.predict(xgb_input)[0]

        # Poisson
        home_p, draw_p, away_p = poisson_match_probs(home_avg, away_avg)
        btts_p = poisson_btts(home_avg, away_avg)

        rows.append({
            "Pred_XGB": pred_xgb,
            "Prob_Poisson_Home": home_p,
            "Prob_Poisson_Draw": draw_p,
            "Prob_Poisson_Away": away_p,
            "Prob_BTTS": btts_p,
            "Label": row["FTR"],
        })

    return pd.DataFrame(rows)

meta_df = build_meta_dataset(data)

meta_features = [
    "Pred_XGB",
    "Prob_Poisson_Home",
    "Prob_Poisson_Draw",
    "Prob_Poisson_Away",
    "Prob_BTTS",
]

y_meta = meta_df["Label"].map({"H": 0, "D": 1, "A": 2})

meta_model = RandomForestClassifier(n_estimators=300, random_state=42)
meta_model.fit(meta_df[meta_features], y_meta)

print("✅ Meta-modelo entrenado")

# -------------------------------
# PREDICCIÓN FINAL
# -------------------------------
def predict_match_final(home, away):
    home_avg = data.groupby("HomeTeam")["FTHG"].mean().get(home, data["FTHG"].mean())
    away_avg = data.groupby("AwayTeam")["FTAG"].mean().get(away, data["FTAG"].mean())

    pred_xgb = model_result.predict(
        np.array([[home_avg, away_avg, home_avg - away_avg]])
    )[0]

    p_home, p_draw, p_away = poisson_match_probs(home_avg, away_avg)
    p_btts = poisson_btts(home_avg, away_avg)

    meta_input = np.array([[pred_xgb, p_home, p_draw, p_away, p_btts]])
    final_pred = meta_model.predict(meta_input)[0]

    mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    return {
        "HomeTeam": home,
        "AwayTeam": away,
        "FinalPrediction": mapping[final_pred],
        "HomeGoals": round(home_avg, 2),
        "AwayGoals": round(away_avg, 2),
        "BTTS_final": "Sí" if p_btts > 0.5 else "No",
        "Prob_BTTS": p_btts,
    }

# -------------------------------
# GENERAR PREDICCIONES DIARIAS
# -------------------------------
matches = [
    ("Real Madrid", "Manchester City"),
    ("Ath Bilbao", "PSG"),
    ("Leverkusen", "Newcastle"),
]

df_pred = pd.DataFrame([predict_match_final(h, a) for h, a in matches])
df_pred.to_csv("data/predictions/last_predictions.csv", index=False)

print("✅ Predicciones generadas")

# -------------------------------
# SUBIR A GITHUB (SOLO EN ACTIONS)
# -------------------------------
if GITHUB_TOKEN != "":
    try:
        auth = Auth.Token(GITHUB_TOKEN)
        g = Github(auth=auth)
        repo = g.get_user().get_repo(GITHUB_REPO)

        remote = "data/predictions/last_predictions.csv"
        with open("data/predictions/last_predictions.csv", "rb") as f:
            content = f.read()

        try:
            existing_file = repo.get_contents(remote)
            repo.update_file(remote, "Daily auto-update", content, existing_file.sha)
        except:
            repo.create_file(remote, "Daily auto-update", content)

        print("✅ Datos subidos a GitHub")

    except Exception as e:
        print("⚠ Error subiendo archivo:", e)

