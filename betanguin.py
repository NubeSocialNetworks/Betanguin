import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from github import Github, Auth

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
GITHUB_TOKEN = os.getenv("NUBE_TOKEN")  # Tu token de GitHub
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

DATA_DIRS = ["data/raw", "data/processed", "data/predictions"]
for d in DATA_DIRS:
    os.makedirs(d, exist_ok=True)

# -------------------------------
# DESCARGA DE DATOS
# -------------------------------
print("📥 Descargando CSVs de ligas...")
for league, url in LEAGUES.items():
    try:
        r = requests.get(url)
        if r.status_code == 200:
            path = f"data/raw/{league}.csv"
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ {league} descargado")
        else:
            print(f"⚠ Error descargando {league}")
    except Exception as e:
        print(f"⚠ Error descargando {league}: {e}")

# -------------------------------
# PROCESAMIENTO Y ENRIQUECIMIENTO
# -------------------------------
print("⚙️ Procesando y enriqueciendo datos...")
all_data = []
for file in os.listdir("data/raw"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/raw/{file}")
        cols_needed = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                       "HY","AY","HR","AR","HC","AC","HST","AST","HF","AF"]
        available_cols = [c for c in cols_needed if c in df.columns]
        all_data.append(df[available_cols])

data = pd.concat(all_data, ignore_index=True)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
data['BTTS'] = ((data['FTHG']>0)&(data['FTAG']>0)).astype(int)
data['HomeGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
data['AwayGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
data['GoalDiff'] = data['HomeGoalsAvg'] - data['AwayGoalsAvg']

# Promedios estadísticos
stats_cols = ['HY','AY','HR','AR','HC','AC','HST','AST','HF','AF']
for t in stats_cols:
    if t in data.columns:
        data[f'{t}_avg'] = data.groupby('HomeTeam')[t].transform('mean') if t.startswith('H') else data.groupby('AwayTeam')[t].transform('mean')

# -------------------------------
# CÁLCULO DE RACHAS
# -------------------------------
def compute_streaks(df, streak_len=5):
    streaks = {}
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in teams:
        team_df = df[(df['HomeTeam']==team)|(df['AwayTeam']==team)].sort_values('Date')
        last_n = team_df.tail(streak_len)
        win_streak, btts_streak, clean_sheet_streak = [], [], []
        for _, row in last_n.iterrows():
            # Win streak
            win_streak.append(1 if ((row['HomeTeam']==team and row['FTR']=='H') or (row['AwayTeam']==team and row['FTR']=='A')) else 0)
            # BTTS streak
            goals_home = row.get('FTHG',0)
            goals_away = row.get('FTAG',0)
            btts_streak.append(1 if goals_home>0 and goals_away>0 else 0)
            # Clean sheet
            clean_sheet_streak.append(1 if (row['HomeTeam']==team and goals_away==0) or (row['AwayTeam']==team and goals_home==0) else 0)
        streaks[team] = {
            'win_streak': sum(win_streak),
            'btts_streak': sum(btts_streak),
            'clean_sheet_streak': sum(clean_sheet_streak)
        }
    return streaks

streaks_dict = compute_streaks(data)
for team_col, prefix in [('HomeTeam','Home'),('AwayTeam','Away')]:
    data[f'{prefix}_Win_Streak'] = data[team_col].map(lambda x: streaks_dict.get(x,{}).get('win_streak',0))
    data[f'{prefix}_BTTS_Streak'] = data[team_col].map(lambda x: streaks_dict.get(x,{}).get('btts_streak',0))
    data[f'{prefix}_CleanSheet_Streak'] = data[team_col].map(lambda x: streaks_dict.get(x,{}).get('clean_sheet_streak',0))

data.to_csv("data/processed/full_dataset_enriched.csv", index=False)
print("✅ Dataset enriquecido guardado")

# -------------------------------
# MODELADO BASE
# -------------------------------
features_base = ['HomeGoalsAvg','AwayGoalsAvg','GoalDiff'] + [f'{t}_avg' for t in stats_cols if f'{t}_avg' in data.columns]
X = data[features_base]
y_result = data['FTR'].map({'H':0,'D':1,'A':2})
y_btts = data['BTTS']

model_result = XGBClassifier(eval_metric='mlogloss')
model_result.fit(X, y_result)

model_btts = XGBClassifier(eval_metric='logloss')
model_btts.fit(X, y_btts)
print("✅ Modelos base entrenados")

# -------------------------------
# FUNCIONES POISSON
# -------------------------------
def poisson_match_probs(home_avg, away_avg, max_goals=10):
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob_matrix[i,j] = poisson.pmf(i, home_avg)*poisson.pmf(j, away_avg)
    return np.sum(np.tril(prob_matrix,-1)), np.sum(np.diag(prob_matrix)), np.sum(np.triu(prob_matrix,1))

def poisson_btts(home_avg, away_avg):
    return (1-np.exp(-home_avg))*(1-np.exp(-away_avg))

# -------------------------------
# META-MODELO
# -------------------------------
meta_features = features_base + ['Home_Win_Streak','Away_Win_Streak','Home_BTTS_Streak','Away_BTTS_Streak','Home_CleanSheet_Streak','Away_CleanSheet_Streak']

def build_meta_dataset(df):
    rows=[]
    for _, row in df.iterrows():
        X_feat = row[features_base].values.reshape(1,-1)  # 🔹 SOLO features_base
        pred_xgb = model_result.predict(X_feat)[0]
        prob_xgb = model_result.predict_proba(X_feat)[0]

        home_avg = row['HomeGoalsAvg']
        away_avg = row['AwayGoalsAvg']
        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_avg, away_avg)
        btts_prob = poisson_btts(home_avg, away_avg)

        rows.append({
            "Pred_XGB": pred_xgb,
            "Prob_XGB_Home": prob_xgb[0],
            "Prob_XGB_Draw": prob_xgb[1],
            "Prob_XGB_Away": prob_xgb[2],
            "Prob_Poisson_Home": home_win_prob,
            "Prob_Poisson_Draw": draw_prob,
            "Prob_Poisson_Away": away_win_prob,
            "Prob_BTTS": btts_prob,
            "HomeGoalsEst": home_avg,
            "AwayGoalsEst": away_avg,
            "BTTS_final": 1 if btts_prob>0.5 else 0,
            "Label": row['FTR']
        })
    return pd.DataFrame(rows)

meta_df = build_meta_dataset(data)
y_meta = meta_df['Label'].map({'H':0,'D':1,'A':2})
meta_model_features = ['Pred_XGB','Prob_XGB_Home','Prob_XGB_Draw','Prob_XGB_Away',
                       'Prob_Poisson_Home','Prob_Poisson_Draw','Prob_Poisson_Away','Prob_BTTS']
meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
meta_model.fit(meta_df[meta_model_features], y_meta)
print("✅ Meta-modelo entrenado")

# -------------------------------
# PREDICCIONES DINÁMICAS CORREGIDAS
# -------------------------------
def predict_matches(matches):
    rows=[]
    for home, away in matches:
        row = data[(data['HomeTeam']==home)&(data['AwayTeam']==away)].tail(1)
        if row.empty:
            row = pd.DataFrame([{
                'HomeTeam':home,'AwayTeam':away,
                'HomeGoalsAvg':data[data['HomeTeam']==home]['HomeGoalsAvg'].mean() if home in data['HomeTeam'].values else data['HomeGoalsAvg'].mean(),
                'AwayGoalsAvg':data[data['AwayTeam']==away]['AwayGoalsAvg'].mean() if away in data['AwayTeam'].values else data['AwayGoalsAvg'].mean(),
                'GoalDiff':0,
                **{f'{t}_avg':0 for t in stats_cols},
                'Home_Win_Streak':0,'Away_Win_Streak':0,
                'Home_BTTS_Streak':0,'Away_BTTS_Streak':0,
                'Home_CleanSheet_Streak':0,'Away_CleanSheet_Streak':0,'FTR':'D'
            }])
        row = row.iloc[0]

        # 🔹 Ajuste dinámico de Poisson según racha
        home_lambda = row['HomeGoalsAvg'] * (1 + 0.1 * row['Home_Win_Streak'])
        away_lambda = row['AwayGoalsAvg'] * (1 + 0.1 * row['Away_Win_Streak'])

        # Predicción XGB
        X_feat = row[features_base].values.reshape(1, -1)
        pred_xgb = model_result.predict(X_feat)[0]
        prob_xgb = model_result.predict_proba(X_feat)[0]

        # Probabilidades Poisson
        home_win_prob, draw_prob, away_win_prob = poisson_match_probs(home_lambda, away_lambda)
        btts_prob = poisson_btts(home_lambda, away_lambda)

        # Meta-modelo
        meta_input = [[pred_xgb, prob_xgb[0], prob_xgb[1], prob_xgb[2],
                       home_win_prob, draw_prob, away_win_prob, btts_prob]]
        final_class = meta_model.predict(meta_input)[0]
        mapping = {0:'Home Win',1:'Draw',2:'Away Win'}

        rows.append({
            "HomeTeam":home,
            "AwayTeam":away,
            "FinalPrediction":mapping[final_class],
            "HomeGoalsEst":round(home_lambda,2),
            "AwayGoalsEst":round(away_lambda,2),
            "BTTS_final":'Sí' if btts_prob>0.5 else 'No',
            "Prob_BTTS":round(btts_prob,3)
        })
    return pd.DataFrame(rows)

# -------------------------------
# EJEMPLO
# -------------------------------
matches = [("Real Madrid","Manchester City"),
           ("Ath Bilbao","PSG"),
           ("Leverkusen","Newcastle"),
           ("Hull", "Wrexham"),
           ("Bristol City", "Leicester City"),
           ("Derby County", "Millwall"),
           ("Ipswich", "Stoke City")]
df_pred = predict_matches(matches)
print(df_pred)
df_pred.to_csv("data/predictions/last_predictions.csv", index=False)

# -------------------------------
# SUBIDA A GITHUB
# -------------------------------
if GITHUB_TOKEN:
    try:
        auth = Auth.Token("NUBE_TOKEN")
        g = Github(auth=auth)
        repo = g.get_user().get_repo("Betanguin")
        remote = "data/predictions/last_predictions.csv"
        with open(remote,"rb") as f:
            content = f.read()
        try:
            existing_file = repo.get_contents(remote)
            repo.update_file(remote,f"Update {remote}",content,existing_file.sha)
        except:
            repo.create_file(remote,f"Add {remote}",content)
        print("✅ Predicción subida a GitHub")
    except Exception as e:
        print(f"⚠ Error subiendo predicción: {e}")

