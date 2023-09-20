import flask
from flask import request
from recommender import pickle_files_saving
import pickle
from utilities import get_top_recommendations
import pandas as pd
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])

def run():

    id = request.args['user_id']

    if any(File.endswith(".pkl") for File in os.listdir(".")):
        # Load SVD model
        with open('svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)

        # Load content-based recommender models
        with open('tfidf_model.pkl', 'rb') as f:
            tfidf = pickle.load(f)

        with open('cosine_sim_model.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
    
    else:
        svd, tfidf, cosine_sim = pickle_files_saving()
        
    df = pd.read_csv("processed.csv")
    df.set_index("Unnamed: 0")
    # define a function to get the top N similar products

    recommendations = get_top_recommendations(df, svd, cosine_sim, customer_id = id, N=10)

    return recommendations

if __name__ == "__main__":
    app.run()