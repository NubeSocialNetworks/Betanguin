def expected(rA, rB):
    return 1 / (1 + 10 ** ((rB - rA) / 400))

def update_rating(r, exp, score, k=20):
    return r + k * (score - exp)

def compute_elo(df):
    teams = list(set(df.home_team.unique()) | set(df.away_team.unique()))
    ratings = {t: 1500 for t in teams}

    elo_h, elo_a = [], []

    for _, m in df.iterrows():
        h, a = m.home_team, m.away_team

        exp_h = expected(ratings[h], ratings[a])
        exp_a = 1 - exp_h

        if m.home_goals > m.away_goals:
            s_h, s_a = 1, 0
        elif m.home_goals < m.away_goals:
            s_h, s_a = 0, 1
        else:
            s_h, s_a = 0.5, 0.5

        ratings[h] = update_rating(ratings[h], exp_h, s_h)
        ratings[a] = update_rating(ratings[a], exp_a, s_a)

        elo_h.append(ratings[h])
        elo_a.append(ratings[a])

    df["elo_home"], df["elo_away"] = elo_h, elo_a
    return df
