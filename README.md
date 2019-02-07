## Toxic Comments Multilabel Classfication
This repo contains code for [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). 
Challenge aims to classify Wikipedia Comments based on toxic behaviour. We train models to classify Wikipedia comments in following 6 types of toxicity:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Average ensembling of model with different hyper parameters and embeddings achieved `0.9860 ROC AUC score` in competition private leaderboard.

## Running Code

#### Setup
This repo uses keras for creating models. Install all dependency using `requirements.txt`. 

`pip install -r requirements.txt`

Download stopwords and punkt tokenizer for nltk.

`python -m nltk.downloader punkt stopwords`

#### Data
Download `train.tsv`, `test.tsv`, `sample_submission.csv` from [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and save in `data` folder.  

Download [fasttext](https://fasttext.cc/docs/en/english-vectors.html) or [glove(840B or twitter)](https://fasttext.cc/docs/en/english-vectors.html) embeddings and save in `data` folder.
 
#### Train Model
Edit configurations in `train_model.py` and train using following command:

`python train_model.py`

#### Test Model
Predict on custom queries  

`python query_model.py --model_iteration 100 --queries "I will kill you" "I hate you"`

