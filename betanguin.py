# ======================================================
# IMPORTS
# ======================================================
import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson

# ======================================================
# CONFIGURACIÓN
# ======================================================
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

stats_cols = ['HY','AY','HR','AR','HC','AC','HST','AST','HF','AF']

# ======================================================
# DESCARGA Y PROCESAMIENTO
# ======================================================
all_data = []
print("📥 Descargando y procesando ligas...")
for league, url in LEAGUES.items():
    try:
        df = pd.read_csv(url)
        cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"] + stats_cols
        use = [c for c in cols if c in df.columns]
        all_data.append(df[use])
        print(f"✔ {league} procesada")
    except Exception as e:
        print(f"⚠ Error en {league}: {e}")

data = pd.concat(all_data, ignore_index=True)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
data['BTTS'] = ((data['FTHG']>0) & (data['FTAG']>0)).astype(int)

# Promedios
data['HomeGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
data['AwayGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
data['GoalDiff'] = data['HomeGoalsAvg'] - data['AwayGoalsAvg']

for t in stats_cols:
    if t in data.columns:
        data[f"{t}_avg"] = data.groupby("HomeTeam")[t].transform('mean') if t.startswith("H") else data.groupby("AwayTeam")[t].transform('mean')

# ======================================================
# CALCULO DE RACHAS
# ======================================================
def compute_streaks(df, streak_len=5):
    streaks = {}
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in teams:
        tdf = df[(df['HomeTeam']==team)|(df['AwayTeam']==team)].sort_values('Date')
        wins = [(1 if ((r['HomeTeam']==team and r['FTR']=='H') or (r['AwayTeam']==team and r['FTR']=='A')) else 0) for _, r in tdf.iterrows()]
        btts = [(1 if r['FTHG']>0 and r['FTAG']>0 else 0) for _, r in tdf.iterrows()]
        streaks[team] = {'win_streak': sum(wins[-streak_len:]), 'btts_streak': sum(btts[-streak_len:])}
    return streaks

streaks_dict = compute_streaks(data)

# ======================================================
# ENTRENAMIENTO MODELOS
# ======================================================
features_base = ['HomeGoalsAvg','AwayGoalsAvg','GoalDiff'] + [f"{t}_avg" for t in stats_cols if f"{t}_avg" in data.columns]

X = data[features_base]
y_result = data['FTR'].map({'H':0,'D':1,'A':2})
y_btts = data['BTTS']

model_result = XGBClassifier(eval_metric='mlogloss')
model_result.fit(X, y_result)

model_btts = XGBClassifier(eval_metric='logloss')
model_btts.fit(X, y_btts)

# ======================================================
# FUNCION DE PREDICCION ROBUSTA
# ======================================================
def predict_matches(matches, data, stats_cols, model_result, model_btts, streaks_dict):
    rows = []

    def get_team_stats(team):
        tdf = data[(data['HomeTeam']==team)|(data['AwayTeam']==team)].sort_values('Date').tail(5)
        if tdf.empty:
            return {
                'HomeGoalsAvg': data['FTHG'].mean(),
                'AwayGoalsAvg': data['FTAG'].mean(),
                **{f"{s}_avg":0 for s in stats_cols},
                'Win_Streak':0,
                'BTTS_Streak':0
            }
        return {
            'HomeGoalsAvg': tdf[tdf['HomeTeam']==team]['FTHG'].mean() if not tdf[tdf['HomeTeam']==team].empty else 0,
            'AwayGoalsAvg': tdf[tdf['AwayTeam']==team]['FTAG'].mean() if not tdf[tdf['AwayTeam']==team].empty else 0,
            **{f"{s}_avg": (tdf[tdf['HomeTeam']==team][s].mean() if s.startswith('H') else tdf[tdf['AwayTeam']==team][s].mean()) for s in stats_cols},
            'Win_Streak': streaks_dict.get(team, {}).get('win_streak',0),
            'BTTS_Streak': streaks_dict.get(team, {}).get('btts_streak',0)
        }

    for home, away in matches:
        try:
            home_stats = get_team_stats(home)
            away_stats = get_team_stats(away)

            X_match = np.array([
                home_stats['HomeGoalsAvg'], away_stats['AwayGoalsAvg'],
                home_stats['HomeGoalsAvg']-away_stats['AwayGoalsAvg']
            ] + [home_stats[f"{s}_avg"] if s.startswith("H") else away_stats[f"{s}_avg"] for s in stats_cols]).reshape(1,-1)

            # Predicción resultado
            prob_res = model_result.predict_proba(X_match)[0]
            prob_res = np.clip(prob_res,0.01,0.99)

            # Predicción BTTS
            prob_btts = model_btts.predict_proba(X_match)[0][1]
            racha_btts_prob = (home_stats['BTTS_Streak'] + away_stats['BTTS_Streak'])/10
            final_btts_prob = 0.6*prob_btts + 0.4*racha_btts_prob
            final_btts_prob = np.clip(final_btts_prob,0.01,0.99)

            mapping = {0:'Home Win',1:'Draw',2:'Away Win'}
            final_class = np.argmax(prob_res)
            final_prediction = mapping[final_class]

            rows.append({
                "HomeTeam": home,
                "AwayTeam": away,
                "FinalPrediction": final_prediction,
                "Prob_HomeWin (%)": round(prob_res[0]*100,2),
                "Prob_Draw (%)": round(prob_res[1]*100,2),
                "Prob_AwayWin (%)": round(prob_res[2]*100,2),
                "Prob_BTTS (%)": round(final_btts_prob*100,2),
                "Prob_BTTS_No (%)": round((1-final_btts_prob)*100,2)
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    return pd.DataFrame(rows)

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
