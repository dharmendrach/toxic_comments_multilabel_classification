import os
import argparse
import numpy as np

import toxic_model.config as config
from toxic_model.models import predict_model
from toxic_model.preprocessing import query_data
from toxic_model.utils import load_pickle, load_json


def predict_query(train_count, query_list):
    """
    model predictions for given queries
    :param train_count: iteration number used for training
    :param query_list: list of text queries for predictions
    :return: labels for queries
    """

    # check if model is trained before predictions
    if not os.path.exists(config.CONFIG_JSON.format(train_count)):
        raise ValueError("Did you trained model first ?")

    # load configurations used in training
    config_json = load_json(config.CONFIG_JSON.format(train_count))

    # load data, We will use only word_index for querying. data pkl can be optimized for query purposes
    data_dict = load_pickle(config.DATA_PKL.format(train_count))

    # convert queries to sequence data
    query_seq_data = query_data(query_list, data_dict['word_index'], config_json['preprocess_config'])

    # check if ensembling was enabled or not
    model_ensemble_count = 1
    if config_json['training_config']['ensemble_kfold']:
        model_ensemble_count = config_json['training_config']['stratified_kfold']

    # predict for all different models checkpoints
    prediction_results = []
    for model_i in range(model_ensemble_count):
        predict_i = predict_model(query_seq_data, config.WEIGHT_FILE.format(train_count, model_i))
        prediction_results.append(predict_i)

    # average ensembling for different models
    predict_output = np.sum(np.asarray(prediction_results), axis=0)/model_ensemble_count

    return predict_output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict toxicity of given comments")
    parser.add_argument("--model_iteration", help="Model directory")
    parser.add_argument("--queries", help="Queries to predict", nargs='+')
    args = parser.parse_args()

    print("Using trained iteration :", args.model_iteration)

    # get predictions for queries
    query_predictions = predict_query(args.model_iteration, args.queries)

    for qi in range(len(args.queries)):
        print("-----------------------------------------")
        print("Query :", args.queries[qi])
        print("Toxic :", query_predictions[qi][0])
        print("Severe Toxic :", query_predictions[qi][1])
        print('Obscene :', query_predictions[qi][2])
        print("Threat :", query_predictions[qi][3])
        print("Insult :", query_predictions[qi][4])
        print("Identity Hate", query_predictions[qi][5])
        print("-----------------------------------------")
