The main folder is main-app.
You can only use the main folder without worrying about the other file


# mentoring-platform-dicoding

## Setup

Start the project with environment setup and run the jupyterlab

```
pip install virtualenv
virtualenv [enviroment name] (misal = ml_mentoring_platform)
source ml_mentoring_platform/bin/activate
pip install -r requirements.txt

```
or run this script for windows users
```
pip install virtualenv
virtualenv [enviroment name]
.\[enviroment name]\Scripts\activate
pip install -r requirements.txt


MODEL : 
https://huggingface.co/abilfad/sentiment-binary-dicoding (TF HF model for sentiment) but stevenliu models are used here because my computer couldn't load tf model 
and also colab's compute units are run out. Actually my model perform better than his, at 93 compared to his in 92
https://huggingface.com/Pudja2001/my_topic_summarizer_model (PyTorch model for summarizer)
Because of l

