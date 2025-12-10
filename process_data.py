import pandas as pd
import glob
import os

def process_file(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result"
    })
    return df[["home_team","away_team","home_goals","away_goals","result"]]

def main():
    os.makedirs("data/processed", exist_ok=True)
    files = glob.glob("data/raw/*.csv")
    dfs = [process_file(f) for f in files]
    full = pd.concat(dfs).reset_index(drop=True)
    full.to_csv("data/processed/matches.csv", index=False)

if __name__ == "__main__":
    main()
