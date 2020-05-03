from flask import Flask, jsonify
from flask_cors import CORS
import LoadModelAndPredict

app = Flask(__name__)


@app.route('/')
def launch():
    print("Yo world.")
    return


def predict_article(article_data):
    """
    Given an article from angular-land, predict on it and return the prediction
    as json.

    :param article_data:
    :return:
    """
    prediction = LoadModelAndPredict.predict(article_data)

    return jsonfiy([prediction])


def index():
    return "Yo dawgs"


if __name__ == '__main__':
    app.run()
