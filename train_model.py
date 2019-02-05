import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit

import toxic_model.config as config
from toxic_model.models import train_model
from toxic_model.preprocessing import prepare_data, get_embedding_matrix
from toxic_model.utils import load_pickle, create_pickle, create_json, custom_multi_label_skfold, create_submission_file

# preprocessing comment config
preprocess_para = {
    "max_seq_len": 100,
    "max_nb_words": 200000,
    "remove_apostrophe": True,
    "remove_stopwords": False,
    "stemming": False,
    "lemmatization": False,
    "alpha_numeric": False
}

# model hyperparameters
model_para = {
    'input_seq_len': preprocess_para['max_seq_len'],  # input sequence length
    'spatial_dropout': 0.2,  # spatial dropout after embedding layer
    'num_rnn_layers': 1,  # number of lstm and gru layers
    'rnn_layer_type': 'gru',  # gru / lstm layer
    'num_rnn_units': [80],  # hidden units for lstm / gru
    'dropout_concat': 0.2,  # dropout before dense layer
    'num_units_dense1': 100,  # hidden units dense layer
    'dropout_dense1': 0.2,  # dropout after dense layer
    'num_output': 6  # number of output labels
}

# training configurations
training_para = {
    'batch_size': 256,
    'max_epoch': 50,
    'shuffle': True,
    'ensemble_kfold': True,  # True for stratified KFold ensembling
    'stratified_kfold': 10,   # number of Folds
    'validation_spit': 0.1,   # will not be used in case of ensemble KFOLD,
                              # floor value of inverse will be used as number of folds
    'embedding_type': 'fasttext_wiki',  # 'glove_840B', 'glove_twitter', 'fasttext_wiki', 'fasttext_crawl'
    'embedding_size': 300,
    'tensorboard_callback': True
}


def prepare_and_train(train_count, configs):
    """
    function for preparing data, training model, test submissions predictions
    :param train_count: train iteration useful for keeping track for different versions
    :param configs: (preprocess_config, training_config, model_para)
    :return: None
    """

    preprocess_config, training_config, model_config = configs

    # creating iteration directory if doesn't exist
    if not os.path.exists(config.OUTPUT_DIR.format(train_count)):
        os.mkdir(config.OUTPUT_DIR.format(train_count))

    # Preprocessing data and create pickle if doesn't exist
    if os.path.exists(config.DATA_PKL.format(train_count)):
        data_dict = load_pickle(config.DATA_PKL.format(train_count))
    else:
        data_dict = prepare_data(config.TRAIN_CSV, config.TEST_CSV, preprocess_config)
        # creating word embedding matrix from word_index dictionary
        data_dict['word_embedding_matrix'] = get_embedding_matrix(training_config['embedding_type'],
                                                                  training_config['embedding_size'],
                                                                  data_dict['word_index'])
        create_pickle(data_dict, config.DATA_PKL.format(train_count))

    # custom Stratified KFold for ensembling
    if training_config['ensemble_kfold']:
        train_val_split = custom_multi_label_skfold(data_dict['train_labels'],
                                                    training_config['stratified_kfold'])
    else:
        # shuffle split for creating train and val split, if not doing ensembling
        shuffle_split = ShuffleSplit(n_splits=1,
                                     test_size=training_config['validation_spit'])
        train_val_split = shuffle_split.split(data_dict['train_labels'])

    test_results = []
    fold_count = 0
    for train_index, val_index in train_val_split:
        # getting train and validation data
        train_data = (data_dict['train_seq'][train_index], data_dict['train_labels'][train_index],
                      data_dict['train_seq'][val_index], data_dict['train_labels'][val_index])

        # training model
        model = train_model(train_count, (training_config, model_config), train_data, fold_count,
                            data_dict['word_embedding_matrix'])

        # predicting on train data and ROC score calculation
        train_predict = model.predict(data_dict['train_seq'])
        train_roc_score = roc_auc_score(data_dict['train_labels'][train_index], train_predict[train_index])
        val_roc_score = roc_auc_score(data_dict['train_labels'][val_index], train_predict[val_index])
        print("train roc score", train_roc_score)
        print("validation roc score", val_roc_score)

        # predicting on test data
        test_results.append(model.predict(data_dict['test_seq']))
        fold_count += 1

    # average ensembling for number for folds
    test_output = np.sum(np.asarray(test_results), axis=0)/fold_count

    # save predictions for submission
    create_submission_file(test_output, config.TEST_SUBMISSION_FILE.format(train_count))

    # save config data in json
    config_data = {
        "timestamp": time.time(),
        "preprocess_config": preprocess_config,
        "model_config": model_config,
        "training_config": training_config
    }
    create_json(config_data, config.CONFIG_JSON.format(train_count))


if __name__ == '__main__':
    # Train and prepare submission for configured hyper parameters
    # weighted ensemble from utils can be used for ensembling different hyper parameters models
    train_iteration = 100
    print("Train iteration :", train_iteration)
    prepare_and_train(train_iteration, (preprocess_para, training_para, model_para))
