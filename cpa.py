import json
import pathlib
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression


def get_calculations(
    exam: str, scores: np.array
) -> Tuple[float, List[float], float, float]:
    """Runs regression or reads it from file"""
    path = pathlib.Path().resolve()

    with open(f"{path}/data/results.json") as file:
        saved_data = json.load(file)

    if f"{exam}{len(scores)}" in saved_data:
        intercept = saved_data[f"{exam}{len(scores)}"]["intercept"]
        coef = saved_data[f"{exam}{len(scores)}"]["coef"]
        mean_inc = saved_data[f"{exam}{len(scores)}"]["mean_inc"]
        std_inc = saved_data[f"{exam}{len(scores)}"]["std_inc"]
        return intercept, coef, mean_inc, std_inc

    df = pd.read_csv(f"{path}/data/{exam}.csv")

    cols = ["B2", "B3"] if len(scores) == 2 else ["B2"]
    x = df[cols]
    y = df["Actual"]

    reg = LinearRegression().fit(x.values, y)
    intercept = reg.intercept_
    coef = reg.coef_

    df["AVG"] = df[cols].mean(axis=1)
    df["INC"] = df["Actual"] - df["AVG"]

    mean_inc = df["INC"].mean()
    std_inc = df["INC"].std()

    saved_data[f"{exam}{len(scores)}"] = {
        "intercept": intercept,
        "coef": coef.tolist(),
        "mean_inc": mean_inc,
        "std_inc": std_inc,
    }

    with open(f"{path}/data/results.json", "w") as f:
        json.dump(saved_data, f)

    return intercept, coef, mean_inc, std_inc


def get_predictions(exam: str, score1: str, score2: str) -> Dict[str, str]:
    """Insert your scores for the two simulated exams for Becker 4.1"""
    if exam not in ["far", "reg", "aud", "bec"]:
        return {"error": "incorrect value for exam"}

    scores = [float(score1), float(score2)] if score2 else [float(score1)]
    np_scores = np.array(scores)
    user_mean = np_scores.mean()

    intercept, coef, mean_inc, std_inc = get_calculations(exam, np_scores)

    z = (mean_inc + user_mean - 75) / std_inc
    prob = st.norm.cdf(z)
    line_pred = intercept + np.dot(scores, coef)
    return {"pred": f"{line_pred:.2f}", "prob": f"{prob*100:.2f}%"}
