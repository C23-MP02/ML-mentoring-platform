import pandas as pd
import numpy as np
import json
from googletrans import Translator 

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)
# model = TFDistilBertForSequenceClassification.from_pretrained(MODEL)

## Function polarity scores
def polarity_scores_roberta(data):
    ### Loading Pre-trained roberta model
    
    encoded_text = tokenizer(data, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative' : scores[0],
        'neutral'  : scores[1],
        'positive' : scores[2]
    }

    max_value = max(scores_dict, key=lambda k: scores_dict[k])
    value     = scores_dict[max_value]
    sentiment = {'Status':max_value,
                 'Value':value}
    return scores_dict , sentiment

def read_data(data='../sentiment-analysis/sample.json'):
    data = data
    with open(data, 'r') as file:
        input = json.load(file)
    input = data
    for dictionary in input:
    # Print the contents of the dictionary
        if 'feedback' in dictionary:
            _ , dictionary['sentiment'] = polarity_scores_roberta(dictionary['feedback'])
    return input

def to_translate(data,dest='en'):
    translator = Translator()
    if isinstance(data, str) == True :
        return translator.translate(data,dest=dest).text
    
    else :
        data_set = data.copy()
        #data_set = pd.DataFrame(data_set)
        translated  = []
        lang_input  = []
        lang_output = []

        for item in data_set['input'] :
            translations = translator.translate(item, dest=dest)
            translated.append(translations.text)
            lang_input.append(translations.src)
            lang_output.append(translations.dest)
            
        data_set['lang_input'] = lang_input
        data_set['translated'] = translated
        data_set['lang_output']= lang_output

        data_pd = pd.DataFrame(data_set)
        return data_pd    

if __name__=="__main__":
    pass

   