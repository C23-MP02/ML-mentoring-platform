from function.rank_similarity import rank_sim
from function.nlp_problem import *
from function.summarizer import *
from flask import Flask, request, json, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hi():
    return 'Success'

@app.route("/cosine_sim", methods=['POST'])
def calc_sim():
    json_ = request.json
    sims = rank_sim(json_)
    return jsonify(sims)

@app.route("/translate",methods=['POST'])
def translate():
    json_      = request.json
    count = 0
    for i in json_:
        temp = json_[count]
        if 'feedback' in temp:
            temp['translate'] = to_translate(dest='en',data=temp['feedback'])
        count +=1
    return jsonify(json_)

@app.route("/sentiment",methods=['POST'])
def sentiment():
    json_      = request.json
    count = 0
    for i in json_:
        if 'feedback' in json_[count]:
            temp = json_[count]
            a = polarity_scores_roberta(temp['feedback'])[1]
            temp['sentiment'] = a['Status']
            count +=1
    return jsonify(json_)

@app.route("/translated-sentiment",methods=['POST'])
def translated_sentiment():
    json_      = request.json
    count = 0
    for i in json_:
        temp = json_[count]
        if 'feedback' in temp:
            temp['translate'] = to_translate(dest='en',data=temp['feedback'])
        count +=1
    
    count = 0
    for i in json_:
        if 'feedback' in json_[count]:
            temp = json_[count]
            a = polarity_scores_roberta(temp['translate'])[1]
            temp['sentiment'] = a['Status']
            count +=1
    
    return jsonify(json_)

@app.route("/feedback_summarizer",methods=['POST'])
def summarize_text():
    json_ = request.json
    summarized_texts = inference_all_data(json_)
    return jsonify(summarized_texts)

if __name__ == '__main__':
    app.run(debug=True, port=5000)