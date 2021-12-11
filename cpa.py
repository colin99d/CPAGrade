from sklearn.linear_model import LinearRegression
import scipy.stats as st
import pandas as pd
import numpy as np

# Insert your scores for the two simulated exams for Becker 4.1
scores = np.array([74,76])

df = pd.read_csv('far.csv')

x = df[["B2","B3"]]
y = df["Actual"]

reg = LinearRegression().fit(x.values, y)
lin_pred = reg.predict(np.array([scores]))
r_score = reg.score(x.values, y)
print(f"Predicted score is {lin_pred[0]:.2f} with an r^2 of {r_score:.2f}")

df["AVG"] = df[['B2', 'B3']].mean(axis = 1)
df["INC"] = df["Actual"] - df["AVG"]

mean_inc = df["INC"].mean()
std = df["INC"].std()
item = scores.mean()

z = (mean_inc + item - 75) / std
prob = st.norm.cdf(z)
print(f"Probability of passing is {prob*100:.2f}%")