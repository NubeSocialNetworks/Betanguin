import pandas as pd
import lightgbm as lgb
import pickle
import os

os.makedirs("data/models", exist_ok=True)

def main():
    df = pd.read_csv("data/processed/final_dataset.csv")

    X = df[["elo_home", "elo_away"]]
    y_home = df["home_goals"]
    y_away = df["away_goals"]

    model_home = lgb.LGBMRegressor()
    model_away = lgb.LGBMRegressor()

    model_home.fit(X, y_home)
    model_away.fit(X, y_away)

    with open("data/models/home.pkl", "wb") as f:
        pickle.dump(model_home, f)
    with open("data/models/away.pkl", "wb") as f:
        pickle.dump(model_away, f)

if __name__ == "__main__":
    main()
