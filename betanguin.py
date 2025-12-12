import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson
from github import Github, Auth

# ======================================================
#  CONFIGURACIÓN
# ======================================================
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

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# ======================================================
#  DESCARGA DE ARCHIVOS
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
            print(f"⚠ Error en {league}")
    except Exception as e:
        print(f"⚠ Error en {league}: {e}")

# ======================================================
#  PROCESAMIENTO Y ENRIQUECIMIENTO
# ======================================================
print("\n⚙️ Procesando y enriqueciendo...")
all_data = []
for file in os.listdir("data/raw"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/raw/{file}")
        cols = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
                "HY","AY","HR","AR","HC","AC","HST","AST","HF","AF"]
        use = [c for c in cols if c in df.columns]
        all_data.append(df[use])

data = pd.concat(all_data, ignore_index=True)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)

# BTTS
data['BTTS'] = ((data['FTHG']>0) & (data['FTAG']>0)).astype(int)
# Promedio goles
data['HomeGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
data['AwayGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
data['GoalDiff'] = data['HomeGoalsAvg'] - data['AwayGoalsAvg']

stats_cols = ['HY','AY','HR','AR','HC','AC','HST','AST','HF','AF']
for t in stats_cols:
    if t in data.columns:
        data[f"{t}_avg"] = data.groupby("HomeTeam")[t].transform('mean') if t.startswith("H") \
                          else data.groupby("AwayTeam")[t].transform('mean')

# ======================================================
#  RACHAS Y FEATURES DERIVADAS
# ======================================================
def compute_streaks_and_reds(df, streak_len=5):
    streaks = {}
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in teams:
        tdf = df[(df['HomeTeam']==team)|(df['AwayTeam']==team)].sort_values("Date")
        wins, btts, clean, red_win, red_draw, red_lose = [], [], [], [], [], []

        for _, r in tdf.iterrows():
            # Win
            wins.append(1 if ((r['HomeTeam']==team and r['FTR']=='H') or
                              (r['AwayTeam']==team and r['FTR']=='A')) else 0)
            # BTTS
            btts.append(1 if r['FTHG']>0 and r['FTAG']>0 else 0)
            # Clean sheet
            clean.append(1 if ((r['HomeTeam']==team and r['FTAG']==0) or
                               (r['AwayTeam']==team and r['FTHG']==0)) else 0)
            # Roja y resultado
            red = 0
            if r['HomeTeam']==team and r.get('HR',0)>0: red=1
            if r['AwayTeam']==team and r.get('AR',0)>0: red=1
            if red:
                if (r['HomeTeam']==team and r['FTR']=='H') or (r['AwayTeam']==team and r['FTR']=='A'):
                    red_win.append(1)
                    red_draw.append(0)
                    red_lose.append(0)
                elif r['FTR']=='D':
                    red_win.append(0)
                    red_draw.append(1)
                    red_lose.append(0)
                else:
                    red_win.append(0)
                    red_draw.append(0)
                    red_lose.append(1)

        streaks[team] = {
            "win_streak": sum(wins[-streak_len:]),
            "btts_streak": sum(btts[-streak_len:]),
            "clean_streak": sum(clean[-streak_len:]),
            "red_win": sum(red_win),
            "red_draw": sum(red_draw),
            "red_lose": sum(red_lose)
        }
    return streaks

st = compute_streaks_and_reds(data)

# ======================================================
#  ENTRENAMIENTO DE MODELOS BASE
# ======================================================
features_base = ['HomeGoalsAvg','AwayGoalsAvg','GoalDiff'] + \
                [f"{t}_avg" for t in stats_cols if f"{t}_avg" in data.columns]

X = data[features_base]
y_result = data['FTR'].map({'H':0,'D':1,'A':2})
y_btts = data['BTTS']

model_result = XGBClassifier(eval_metric='mlogloss')
model_result.fit(X, y_result)

model_btts = XGBClassifier(eval_metric='logloss')
model_btts.fit(X, y_btts)

# ======================================================
#  FUNCIONES POISSON
# ======================================================
def poisson_match_probs(h,a,max_goals=10):
    M = np.zeros((max_goals+1,max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            M[i,j] = poisson.pmf(i,h)*poisson.pmf(j,a)
    return M.sum(axis=1).sum()-np.trace(M), np.trace(M), M.sum()-M.sum(axis=1).sum()

def poisson_btts(h,a):
    return (1-np.exp(-h))*(1-np.exp(-a))

# ======================================================
#  META-MODELO
# ======================================================
meta_rows = []
for _, r in data.iterrows():
    X_feat = r[features_base].values.reshape(1,-1)
    pred_xgb = model_result.predict(X_feat)[0]
    prob_xgb = model_result.predict_proba(X_feat)[0]

    home_avg = r['HomeGoalsAvg']
    away_avg = r['AwayGoalsAvg']

    ph, pd_draw, pa = poisson_match_probs(home_avg, away_avg)
    pb = poisson_btts(home_avg, away_avg)

    meta_rows.append([pred_xgb, prob_xgb[0], prob_xgb[1], prob_xgb[2],
                      ph, pd_draw, pa, pb, r['FTR']])

meta_df = pd.DataFrame(meta_rows, columns=[
    "Pred_XGB","Prob_XGB_Home","Prob_XGB_Draw","Prob_XGB_Away",
    "Prob_Poisson_Home","Prob_Poisson_Draw","Prob_Poisson_Away","Prob_BTTS","Label"
])
meta_df["Label"] = meta_df["Label"].map({'H':0,'D':1,'A':2})

meta_model = RandomForestClassifier(n_estimators=200, random_state=42)
meta_model.fit(meta_df.drop(columns=["Label"]), meta_df["Label"])

# ======================================================
#  FUNCIÓN FINAL DE PREDICCIÓN
# ======================================================
def get_team_stats(team, data, stats_cols, streaks_dict):
    home_goals_avg = data[data['HomeTeam']==team]['FTHG'].mean() if team in data['HomeTeam'].values else data['FTHG'].mean()
    away_goals_avg = data[data['AwayTeam']==team]['FTAG'].mean() if team in data['AwayTeam'].values else data['FTAG'].mean()
    stat_avgs = {}
    for t in stats_cols:
        if t.startswith('H'):
            stat_avgs[f'{t}_avg'] = data[data['HomeTeam']==team][t].mean() if team in data['HomeTeam'].values else 0
        else:
            stat_avgs[f'{t}_avg'] = data[data['AwayTeam']==team][t].mean() if team in data['AwayTeam'].values else 0
    streaks = streaks_dict.get(team, {'win_streak':0,'btts_streak':0,'clean_streak':0})
    return {
        'HomeGoalsAvg': home_goals_avg,
        'AwayGoalsAvg': away_goals_avg,
        **stat_avgs,
        'Win_Streak': streaks['win_streak'],
        'BTTS_Streak': streaks['btts_streak'],
        'Clean_Streak': streaks['clean_streak']
    }

def predict_matches(matches):
    rows = []
    for home, away in matches:
        home_stats = get_team_stats(home, data, stats_cols, st)
        away_stats = get_team_stats(away, data, stats_cols, st)

        # Ajuste dinámico Poisson
        home_lambda = home_stats['HomeGoalsAvg']*(1+0.1*home_stats['Win_Streak'])
        away_lambda = away_stats['AwayGoalsAvg']*(1+0.1*away_stats['Win_Streak'])

        # Features vector
        feature_vector = np.array([
            home_stats['HomeGoalsAvg'], away_stats['AwayGoalsAvg'],
            home_stats['HomeGoalsAvg']-away_stats['AwayGoalsAvg']
        ] + [home_stats.get(f'{t}_avg',0) for t in stats_cols if t.startswith('H')] +
            [away_stats.get(f'{t}_avg',0) for t in stats_cols if t.startswith('A')]
        ).reshape(1,-1)

        # Predicción XGB
        pred_xgb = model_result.predict(feature_vector)[0]
        prob_xgb = model_result.predict_proba(feature_vector)[0]

        # Poisson
        ph, pd_draw, pa = poisson_match_probs(home_lambda, away_lambda)
        pb = poisson_btts(home_lambda, away_lambda)

        meta_input = [[pred_xgb, prob_xgb[0], prob_xgb[1], prob_xgb[2], ph, pd_draw, pa, pb]]
        final_class = meta_model.predict(meta_input)[0]
        mapping = {0:'Home Win',1:'Draw',2:'Away Win'}
        final_prediction = mapping[final_class]

        # BTTS ajustado por racha
        racha_btts_prob = (home_stats['BTTS_Streak']+away_stats['BTTS_Streak'])/10
        final_btts_prob = 0.5*racha_btts_prob + 0.5*pb

        rows.append({
            'HomeTeam': home,
            'AwayTeam': away,
            'FinalPrediction': final_prediction,
            'HomeGoalsEst': round(home_lambda,2),
            'AwayGoalsEst': round(away_lambda,2),
            'BTTS_final': 'Sí' if final_btts_prob>0.5 else 'No',
            'Prob_BTTS': round(final_btts_prob,3)
        })
    return pd.DataFrame(rows)

# ======================================================
#  EJEMPLO DE USO
# ======================================================
#matches = [
#    ("Angers","Nantes"), ("Nancy","Clermont"), ("Pau","Amiens"),
#    ("Laval","Dunkerque"), ("West Brom","Sheffield United"),
#    ("Union Berlin","RB Leipzig"), ("Greuther Furth","Hertha"),
#    ("Dresden","Braunschweig"), ("Sociedad","Girona"),
#    ("Cultural Leonesa","Huesca"), ("Lecce","Pisa")
#]
#
#df_pred = predict_matches(matches)
#print(df_pred)

