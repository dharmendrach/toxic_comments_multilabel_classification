import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import toxic_model.config as config


def create_pickle(data, filename):
    """
    Create pickle for given data
    :param data: data object to be saved
    :param filename: pickle file name
    :return: None
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filename):
    """
    Load saved data from pickle file
    :param filename: pickle file name
    :return: data from pickle file
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_json(filename):
    """
    loading json file
    :param filename: json file name
    :return: data as dict from json file
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def create_json(data, filename):
    """
    saving data in json file
    :param data: data to be saved
    :param filename:  json file name
    :return: None
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def create_submission_file(predicted_labels, output_file):
    """
    create file for submission
    :param predicted_labels: predicted label for test data
    :param output_file: output file path
    :return: None
    """
    sample_submission = pd.read_csv(config.SUBMISSION_FILE)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    sample_submission[list_classes] = predicted_labels
    # saving pandas dataframe as csv
    sample_submission.to_csv(output_file, index=False)


def custom_multi_label_skfold(labels, n_folds, random_state=42):
    """

    :param labels: lablels
    :param n_folds: number of folds
    :param random_state: random state for StratifiedKFold
    :return: generator for K Folds, will return train_index and val_index
    """
    # We will convert 6 toxicity labels to (0,63)
    c = np.asarray([1, 2, 4, 8, 16, 32])
    d = np.dot(labels, c)

    # apply stratifiedKFold on converted labels
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    # return generator for train and validation index
    return ((train_index, val_index) for train_index, val_index in skf.split(d, d))


def weighted_ensemble(output_files_list, weights, submission_file):
    """
    Weighted ensemble for predictions
    :param output_files_list: list of output submission file
    :param weights: weightage for output submission file predictions
    :param submission_file: ensemble output file
    :return: None
    """
    predictions = 0
    for file_i in range(len(output_files_list)):
        output_df = pd.read_csv(output_files_list[file_i])
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        predictions += weights[file_i] * output_df[list_classes].values
    predictions /= np.sum(weights)
    create_submission_file(predictions, submission_file)
