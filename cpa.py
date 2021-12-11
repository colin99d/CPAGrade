import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression


def get_predictions(score1: str, score2: str):
    """Insert your scores for the two simulated exams for Becker 4.1"""
    scores = np.array([float(score1), float(score2)])

    df = pd.read_csv("far.csv")

    x = df[["B2", "B3"]]
    y = df["Actual"]

    reg = LinearRegression().fit(x.values, y)
    lin_pred = reg.predict(np.array([scores]))
    r_score = reg.score(x.values, y)

    df["AVG"] = df[["B2", "B3"]].mean(axis=1)
    df["INC"] = df["Actual"] - df["AVG"]

    mean_inc = df["INC"].mean()
    std = df["INC"].std()
    item = scores.mean()

    z = (mean_inc + item - 75) / std
    prob = st.norm.cdf(z)
    return {"pred": lin_pred[0], "r2": r_score, "prob": prob}
