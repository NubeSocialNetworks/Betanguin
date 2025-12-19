# ======================================================
# IMPORTS
# ======================================================
import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import poisson
from github import Github

# ======================================================
# CONFIGURACIÓN
# ======================================================
GITHUB_TOKEN = os.getenv("NUBE_TOKEN")  # Token GitHub
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

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# ======================================================
# DESCARGA DE ARCHIVOS
# ======================================================
print("📥 Descargando ligas...")
for league, url in LEAGUES.items():
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"data/raw/{league}.csv", "wb") as f:
                f.write(resp.content)
            print(f"✔ {league} descargada")
        else:
            print(f"⚠ Error descargando {league}")
    except Exception as e:
        print(f"⚠ Error {league}: {e}")

# ======================================================
# PROCESAMIENTO Y ENRIQUECIMIENTO
# ======================================================
print("⚙️ Procesando datos...")
all_data = []
stats_cols = ['HY','AY','HR','AR','HC','AC','HST','AST','HF','AF']

for file in os.listdir("data/raw"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/raw/{file}")
        cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"] + stats_cols
        use = [c for c in cols if c in df.columns]
        all_data.append(df[use])

data = pd.concat(all_data, ignore_index=True)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
data = data.sort_values('Date')

# BTTS
data['BTTS'] = ((data['FTHG']>0) & (data['FTAG']>0)).astype(int)

# Promedios xG últimos 5 partidos
data['xG_Home_Last5'] = data.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift().rolling(5).mean())
data['xG_Away_Last5'] = data.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift().rolling(5).mean())
data['xG_Home_Last5'] = data['xG_Home_Last5'].fillna(data['xG_Home_Last5'].mean())
data['xG_Away_Last5'] = data['xG_Away_Last5'].fillna(data['xG_Away_Last5'].mean())

# Stats últimos 5 partidos
for s in stats_cols:
    data[f"{s}_Last5"] = data.groupby('HomeTeam')[s].transform(lambda x: x.shift().rolling(5).mean() if s.startswith('H') else x.shift().rolling(5).mean())
    data[f"{s}_Last5"] = data[f"{s}_Last5"].fillna(data[s].mean())

# ======================================================
# ENTRENAMIENTO MODELOS
# ======================================================
features = ['xG_Home_Last5','xG_Away_Last5'] + [f"{s}_Last5" for s in stats_cols]
X = data[features].fillna(0)
y_result = data['FTR'].map({'H':0,'D':1,'A':2})
y_btts = data['BTTS']

model_result = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
model_result.fit(X, y_result)

model_btts = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_btts.fit(X, y_btts)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def get_team_stats_recent(team, data, stats_cols):
    tdf = data[(data['HomeTeam']==team) | (data['AwayTeam']==team)].sort_values("Date").tail(5)
    if tdf.empty:
        return None
    return {
        'xG_Home_Last5': tdf[tdf['HomeTeam']==team]['FTHG'].mean() if not tdf[tdf['HomeTeam']==team].empty else 0,
        'xG_Away_Last5': tdf[tdf['AwayTeam']==team]['FTAG'].mean() if not tdf[tdf['AwayTeam']==team].empty else 0,
        **{f"{s}_Last5": (tdf[tdf['HomeTeam']==team][s].mean() if s.startswith('H') else tdf[tdf['AwayTeam']==team][s].mean()) for s in stats_cols}
    }

def kelly_stake(prob, odd, fraction=0.5):
    if odd is None or odd <= 1:
        return 0
    edge = prob - 1/odd
    kelly = edge / (odd - 1)
    return round(max(kelly * fraction * 100, 0),2)

# ======================================================
# CALCULAR STAKES
# ======================================================
def calculate_stakes(matches, odds_dict, kelly_fraction=0.5):
    output = []
    for home, away in matches:
        stats_home = get_team_stats_recent(home, data, stats_cols)
        stats_away = get_team_stats_recent(away, data, stats_cols)
        if stats_home is None or stats_away is None:
            continue

        X_match = np.array([
            stats_home['xG_Home_Last5'], stats_away['xG_Away_Last5']
        ] + [stats_home[f"{s}_Last5"] if s.startswith("H") else stats_away[f"{s}_Last5"] for s in stats_cols]).reshape(1,-1)

        prob_res = model_result.predict_proba(X_match)[0]
        prob_btts = model_btts.predict_proba(X_match)[0][1]
        prob_btts_no = 1 - prob_btts

        odds = odds_dict.get((home,away), {})

        output.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "Prob_H (%)": round(prob_res[0]*100,2),
            "Prob_D (%)": round(prob_res[1]*100,2),
            "Prob_A (%)": round(prob_res[2]*100,2),
            "Stake_H (%)": kelly_stake(prob_res[0], odds.get("H"), kelly_fraction),
            "Stake_D (%)": kelly_stake(prob_res[1], odds.get("D"), kelly_fraction),
            "Stake_A (%)": kelly_stake(prob_res[2], odds.get("A"), kelly_fraction),
            "Stake_BTTS_Yes (%)": kelly_stake(prob_btts, odds.get("BTTS_Sí"), kelly_fraction),
            "Stake_BTTS_No (%)": kelly_stake(prob_btts_no, odds.get("BTTS_No"), kelly_fraction)
        })
    return pd.DataFrame(output)

# ======================================================
# ACTUALIZAR CSV ENRIQUECIDO Y SUBIR A GITHUB
# ======================================================
ENRICHED_CSV_PATH = "data/processed/full_dataset_enriched.csv"
data.to_csv(ENRICHED_CSV_PATH, index=False)
print(f"✔ CSV actualizado: {ENRICHED_CSV_PATH}")

# Subida automática a GitHub
try:
    g = Github(GITHUB_TOKEN)
    repo = g.get_user(GITHUB_USER).get_repo(GITHUB_REPO)
    with open(ENRICHED_CSV_PATH, "r") as f:
        content = f.read()
    file = repo.get_contents("data/processed/full_dataset_enriched.csv")
    repo.update_file(file.path, "Auto-update enriched CSV", content, file.sha)
    print("✔ CSV subido a GitHub correctamente")
except Exception as e:
    print(f"⚠ Error subiendo CSV a GitHub: {e}")
