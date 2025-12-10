def add_features(df):
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    return df
