import pandas as pd
import numpy as np
import json
from googletrans import Translator 

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification,TFAutoModelForSequenceClassification
from scipy.special import softmax
#import tensorflow as tf
import torch


# tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)
# model = TFDistilBertForSequenceClassification.from_pretrained(MODEL)

## Function polarity scores
# def polarity_scores_roberta(data):
#     ### Loading Pre-trained roberta model
#     MODEL = f"cardiffnlp/roberta-base-sentiment"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#     encoded_text = tokenizer(data, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dict = {
#         'negative' : scores[0],
#         'neutral'  : scores[1],
#         'positive' : scores[2]
#     }

#     max_value = max(scores_dict, key=lambda k: scores_dict[k])
#     value     = scores_dict[max_value]
#     sentiment = {'Status':max_value,
#                  'Value':value}
#     return scores_dict , sentiment

def binary_score_abil_dicoding(text):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    # MODEL = f"abilfad/sentiment-binary-dicoding"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # inputs = tokenizer(text, return_tensors="tf") # pt for pytorch | tf for tensorflow
    #### TF MODEL ####
    # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL,id2label=id2label, label2id=label2id)
    # logits = model(**inputs).logits
    # predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    
    ### TORCH MODEL ####
    MODEL_TORCH = f"stevhliu/my_awesome_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TORCH)
    inputs = tokenizer(text, return_tensors="tf")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TORCH, id2label=id2label, label2id=label2id)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    
    
    return str(model.config.id2label[predicted_class_id]).lower()

# def read_data(data='../sentiment-analysis/sample.json'):
#     data = data
#     with open(data, 'r') as file:
#         input = json.load(file)
#     input = data
#     for dictionary in input:
#     # Print the contents of the dictionary
#         if 'feedback' in dictionary:
#             _ , dictionary['sentiment'] = polarity_scores_roberta(dictionary['feedback'])
#     return input

# def to_translate(data,dest='en'):
#     translator = Translator()
#     if isinstance(data, str) == True :
#         return translator.translate(data,dest=dest).text
    
#     else :
#         data_set = data.copy()
#         #data_set = pd.DataFrame(data_set)
#         translated  = []
#         lang_input  = []
#         lang_output = []

#         for item in data_set['input'] :
#             translations = translator.translate(item, dest=dest)
#             translated.append(translations.text)
#             lang_input.append(translations.src)
#             lang_output.append(translations.dest)
            
#         data_set['lang_input'] = lang_input
#         data_set['translated'] = translated
#         data_set['lang_output']= lang_output

#         data_pd = pd.DataFrame(data_set)
#         return data_pd
     
def to_translate(data, dest='en'):
    translator = Translator()
    if isinstance(data, str):
        return translator.translate(data, dest=dest).text
    else:
        data_set = data.copy()

        translations = [translator.translate(item, dest=dest) for item in data_set['input']]

        data_set['lang_input'] = [trans.src for trans in translations]
        data_set['translated'] = [trans.text for trans in translations]
        data_set['lang_output'] = [trans.dest for trans in translations]

        return pd.DataFrame(data_set)
   

if __name__=="__main__":
    pass
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    # MODEL = f"abilfad/sentiment-binary-dicoding"
    # text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # inputs = tokenizer(text, return_tensors="pt")

    # model = AutoModelForSequenceClassification.from_pretrained(MODEL,id2label=id2label, label2id=label2id)
    # with torch.no_grad():
    #     logits = model(**inputs).logits
    # predicted_class_id = logits.argmax().item()
    # print(str(model.config.id2label[predicted_class_id]).lower())
    

   