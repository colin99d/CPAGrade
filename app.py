from cpa import get_predictions
from dotenv import load_dotenv
from flask import Flask
from flask import render_template
from flask import request

load_dotenv()

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        score1 = request.form.get("score1", None)
        score2 = request.form.get("score2", None)
        exam = request.form.get("exam", None)
        pred_vals = get_predictions(exam, score1, score2)
        return render_template("main.html", result=pred_vals)
    return render_template("main.html", result=None)
