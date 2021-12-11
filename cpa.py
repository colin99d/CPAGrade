import pathlib

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression


def get_predictions(exam: str, score1: str, score2: str):
    """Insert your scores for the two simulated exams for Becker 4.1"""
    scores = (
        np.array([float(score1), float(score2)])
        if score2
        else np.array([float(score1)])
    )

    if exam not in ["far", "reg", "aud", "bec"]:
        return {"error": "incorrect value for exam"}

    path = pathlib.Path().resolve()
    df = pd.read_csv(f"{path}/data/{exam}.csv")

    cols = ["B2", "B3"] if score2 else ["B2"]
    x = df[cols]
    y = df["Actual"]

    reg = LinearRegression().fit(x.values, y)
    lin_pred = reg.predict(np.array([scores]))

    df["AVG"] = df[cols].mean(axis=1)
    df["INC"] = df["Actual"] - df["AVG"]

    mean_inc = df["INC"].mean()
    std = df["INC"].std()
    item = scores.mean()

    z = (mean_inc + item - 75) / std
    prob = st.norm.cdf(z)
    return {"pred": f"{lin_pred[0]:.2f}", "prob": f"{prob*100:.2f}%"}
