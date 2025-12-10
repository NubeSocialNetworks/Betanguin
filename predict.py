import pandas as pd
import pickle

def predict_match(home_elo, away_elo):
    with open("data/models/home.pkl", "rb") as f:
        m_home = pickle.load(f)
    with open("data/models/away.pkl", "rb") as f:
        m_away = pickle.load(f)

    X = pd.DataFrame([[home_elo, away_elo]], columns=["elo_home","elo_away"])
    return {
        "pred_home_goals": float(m_home.predict(X)),
        "pred_away_goals": float(m_away.predict(X))
    }
