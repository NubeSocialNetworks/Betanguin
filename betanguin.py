import os, requests, pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from github import Github, Auth

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
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
# PROCESAMIENTO DE DATOS
# -------------------------------
all_data = []
for file in os.listdir("data/raw"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/raw/{file}")
        cols_needed = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HY","AY","HR","AR"]
        available_cols = [c for c in cols_needed if c in df.columns]
        all_data.append(df[available_cols])
data = pd.concat(all_data, ignore_index=True)

# Parse dates
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)

# BTTS histórico
data['BTTS'] = ((data['FTHG'] > 0) & (data['FTAG'] > 0)).astype(int)

# Promedios
data['HomeGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
data['AwayGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
data['GoalDiff'] = data['HomeGoalsAvg'] - data['AwayGoalsAvg']

# Promedio de tarjetas
for t in ['HY','AY','HR','AR']:
    if t in data.columns:
        data[f'{t}_avg'] = data.groupby('HomeTeam')[t].transform('mean') if t.startswith('H') else data.groupby('AwayTeam')[t].transform('mean')

data.to_csv("data/processed/full_dataset_enriched.csv", index=False)
print("✅ Dataset procesado y enriquecido")

# -------------------------------
# MODELOS BASE
# -------------------------------
features = ['HomeGoalsAvg','AwayGoalsAvg','GoalDiff'] + [f'{t}_avg' for t in ['HY','AY','HR','AR'] if f'{t}_avg' in data.columns]

X = data[features]
y_result = data['FTR'].map({'H':0,'D':1,'A':2})
y_btts = data['BTTS']

# XGB
model_result = XGBClassifier(eval_metric='mlogloss')
model_result.fit(X, y_result)

model_btts = XGBClassifier(eval_metric='logloss')
model_btts.fit(X, y_btts)
print("✅ Modelos entrenados")

# -------------------------------
# POISSON
# -------------------------------
def poisson_match_probs(home_avg, away_avg, max_goals=10):
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob_matrix[i,j] = poisson.pmf(i, home_avg) * poisson.pmf(j, away_avg)
    home_win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    away_win_prob = np.sum(np.triu(prob_matrix, 1))
    return home_win_prob, draw_prob, away_win_prob

def poisson_btts(home_avg, away_avg):
    prob_home_scores = 1 - np.exp(-home_avg)
    prob_away_scores = 1 - np.exp(-away_avg)
    return prob_home_scores * prob_away_scores

# -------------------------------
# META-MODELO
# -------------------------------
def build_meta_dataset(data):
    rows = []
    for idx, row in data.iterrows():
        home_avg = row.get('HomeGoalsAvg', 0)
        away_avg = row.get('AwayGoalsAvg', 0)
        goal_diff = row.get('GoalDiff', 0)
        HY_avg = row.get('HY_avg', 0)
        AY_avg = row.get('AY_avg', 0)
        HR_avg = row.get('HR_avg', 0)
        AR_avg = row.get('AR_avg', 0)

        xgb_features = np.array([[home_avg, away_avg, goal_diff, HY_avg, AY_avg, HR_avg, AR_avg]])
        pred_xgb = model_result.predict(xgb_features)[0]
        prob_xgb = model_result.predict_proba(xgb_features)[0]

        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_avg, away_avg)
        btts_prob = poisson_btts(home_avg, away_avg)
        label = row.get('FTR','D')

        rows.append({
            "Pred_XGB": pred_xgb,
            "Prob_XGB_Home": prob_xgb[0],
            "Prob_XGB_Draw": prob_xgb[1],
            "Prob_XGB_Away": prob_xgb[2],
            "Prob_Poisson_Home": home_win_prob,
            "Prob_Poisson_Draw": draw_prob,
            "Prob_Poisson_Away": away_win_prob,
            "Prob_BTTS": btts_prob,
            "Label": label
        })
    return pd.DataFrame(rows)

meta_df = build_meta_dataset(data)
meta_features = ['Pred_XGB','Prob_XGB_Home','Prob_XGB_Draw','Prob_XGB_Away',
                 'Prob_Poisson_Home','Prob_Poisson_Draw','Prob_Poisson_Away','Prob_BTTS']
y_meta = meta_df['Label'].map({'H':0,'D':1,'A':2})

meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
meta_model.fit(meta_df[meta_features], y_meta)
print("✅ Meta-modelo entrenado")

# -------------------------------
# PREDICCIONES
# -------------------------------
def show_prediction_table_final(matches):
    rows = []
    for home, away in matches:
        home_avg = data.groupby('HomeTeam')['FTHG'].mean().get(home, data['FTHG'].mean())
        away_avg = data.groupby('AwayTeam')['FTAG'].mean().get(away, data['FTAG'].mean())
        goal_diff = home_avg - away_avg
        HY_avg = data.groupby('HomeTeam')['HY'].mean().get(home,0) if 'HY' in data.columns else 0
        AY_avg = data.groupby('AwayTeam')['AY'].mean().get(away,0) if 'AY' in data.columns else 0
        HR_avg = data.groupby('HomeTeam')['HR'].mean().get(home,0) if 'HR' in data.columns else 0
        AR_avg = data.groupby('AwayTeam')['AR'].mean().get(away,0) if 'AR' in data.columns else 0

        xgb_features = np.array([[home_avg, away_avg, goal_diff, HY_avg, AY_avg, HR_avg, AR_avg]])
        pred_xgb = model_result.predict(xgb_features)[0]
        prob_xgb = model_result.predict_proba(xgb_features)[0]

        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_avg, away_avg)
        btts_prob = poisson_btts(home_avg, away_avg)

        meta_input = np.array([[pred_xgb, prob_xgb[0], prob_xgb[1], prob_xgb[2],
                                home_win_prob, draw_prob, away_win_prob, btts_prob]])
        final_pred_class = meta_model.predict(meta_input)[0]
        mapping = {0:'Home Win',1:'Draw',2:'Away Win'}
        btts_final = 1 if btts_prob > 0.5 else 0

        rows.append({
            "HomeTeam": home,
            "AwayTeam": away,
            "FinalPrediction": mapping[final_pred_class],
            "HomeGoals": home_avg,
            "AwayGoals": away_avg,
            "BTTS_final": 'Sí' if btts_final else 'No',
            "Prob_BTTS": btts_prob
        })
    return pd.DataFrame(rows)

# -------------------------------
# EJEMPLO USO
# -------------------------------
matches = [("Real Madrid","Manchester City"),
           ("Ath Bilbao","PSG"),
           ("Leverkusen","Newcastle"),
           ("Hull", "Wrexham"),
           ("Bristol City", "Leicester City"),
           ("Derby County", "Millwall"),
           ("Ipswich", "Stoke City")]

df_pred = show_prediction_table_final(matches)
print(df_pred)

# Guardar predicción
df_pred.to_csv("data/predictions/last_predictions.csv", index=False)

# Subir a GitHub
try:
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_user().get_repo(GITHUB_REPO)
    remote = "data/predictions/last_predictions.csv"
    with open("data/predictions/last_predictions.csv","rb") as f:
        content = f.read()
    try:
        existing_file = repo.get_contents(remote)
        repo.update_file(remote, f"Update {remote}", content, existing_file.sha)
    except:
        repo.create_file(remote, f"Add {remote}", content)
    print("✅ Predicción subida a GitHub")
except Exception as e:
    print(f"⚠ Error subiendo predicción: {e}")

