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


@app.route("/translated-sentiment",methods=['POST'])
def translated_sentiment():
    json_ = request.json
    for item in json_:
        if 'feedback' in item:
            item['translate'] = to_translate(dest='en',data=item['feedback'])
            item['sentiment'] = binary_score_abil_dicoding(item['translate'])
    
    return jsonify(json_)

@app.route("/translate-en-id",methods=['POST'])
def translated_sentiment_en_id():
    json_ = request.json
    for item in json_:
        if 'feedback' in item:
            item['translate'] = to_translate(dest='id',data=item['feedback'])
    
    return jsonify(json_)


@app.route("/feedback_summarizer",methods=['POST'])
def summarize_text():
    json_ = request.json
    summarized_texts = inference_all_data(json_)
    return jsonify(summarized_texts)

@app.route("/feedback_summarizer_id",methods=['POST'])
def summarize_text_id():
    json_ = request.json
    summarized_texts = inference_all_data(json_)
    feedback_list = summarized_texts["feedback"]
    if len(feedback_list) > 0:  # check that there's at least one item in the list
        feedback_text = feedback_list[0]  # get the first item
        feedback_dict = {'feedback': feedback_text}  # create a dictionary
        id_summarized = to_translate(dest='id', data=[feedback_dict])  # note the list around feedback_dict
    else:
        id_summarized = {'error': 'No feedback to translate'}
    return jsonify(id_summarized)


# @app.route("/sentiment",methods=['POST'])
# def sentiment():
#     json_      = request.json
#     count = 0
#     for i in json_:
#         if 'feedback' in json_[count]:
#             temp = json_[count]
#             a = polarity_scores_roberta(temp['feedback'])[1]
#             temp['sentiment'] = a['Status']
#             count +=1
#     return jsonify(json_)


if __name__ == '__main__':
    app.run(debug=True, port=5000)