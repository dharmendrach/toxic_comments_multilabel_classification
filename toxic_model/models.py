from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, GlobalAveragePooling1D,\
    GlobalMaxPooling1D, LSTM, GRU, Bidirectional, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import toxic_model.config as config


def rnn_model(model_config, word_embedding_matrix):
    """
    Building model graph for given model config
    :param model_config: configurations and hyperparameters for model
    :param word_embedding_matrix: embedding matrix created using pretrained vectors
    :return: model graph
    """
    # input layer with
    inp = Input(shape=(model_config['input_seq_len'],))

    # embedding layer, we will only use pretrained vectors as data is not enough for training
    x = Embedding(word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                  trainable=False)(inp)
    # spatial dropout
    x = SpatialDropout1D(model_config['spatial_dropout'])(x)

    # bidirectional lstm or gru layers with return sequence true argument
    for rnn_layer_i in range(model_config['num_rnn_layers']):
        if model_config['rnn_layer_type'] == 'lstm':
            x = Bidirectional(LSTM(model_config['num_rnn_units'][rnn_layer_i], return_sequences=True))(x)
        else:
            x = Bidirectional(GRU(model_config['num_rnn_units'][rnn_layer_i], return_sequences=True))(x)

    # GlobalMaxPooling and GloablAverage Pooling to convert data to batch_size, num_rnn_units
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concat = concatenate([avg_pool, max_pool])  # concat of GlobalMax and GlobalAverage pooling

    concat = Dropout(model_config['dropout_concat'])(concat)  # dropout layer
    concat = BatchNormalization()(concat)  # batch normalization

    # Dense layer with relu activation, followed by dropout and batch normalization layer
    concat = Dense(model_config['num_units_dense1'], activation="relu")(concat)
    concat = Dropout(model_config['dropout_dense1'])(concat)
    concat = BatchNormalization()(concat)

    # output layer with sigmoid activation
    out = Dense(model_config['num_output'], activation="sigmoid")(concat)
    model = Model(inputs=inp, outputs=out)

    # printing model summary
    model.summary()
    # binary cross entropy loss and adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(train_count, configs, train_data, fold_i, word_embedding_matrix):
    """
    Creating model and training
    :param train_count: train iteration
    :param configs: (training_config, model_config)
    :param train_data: (train_seq, train_labels, val_seq, val_labels)
    :param fold_i: fold index, useful for ensembling
    :param word_embedding_matrix: pretrained word embedding matrix for given data
    :return: trained model
    """
    training_config, model_config = configs
    (train_seq, train_label, val_seq, val_label) = train_data

    # creating lstm / gru models
    model = rnn_model(model_config, word_embedding_matrix)

    # model save checkpoint and early stopping callbacks
    checkpoint = ModelCheckpoint(config.WEIGHT_FILE.format(train_count, fold_i), monitor='val_loss',
                                 verbose=2, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callback_list = [checkpoint, early]

    # for model debugging purpose tensorboard callback
    if training_config['tensorboard_callback']:
        tensorboard_call = TensorBoard(log_dir=config.OUTPUT_DIR.format(train_count), histogram_freq=0,
                                       write_graph=True, write_images=True)
        callback_list.append(tensorboard_call)

    # training model on given data
    model.fit(train_seq, train_label, batch_size=training_config['batch_size'],
              epochs=training_config['max_epoch'],
              validation_data=(val_seq, val_label), shuffle=training_config['shuffle'],
              callbacks=callback_list)
    return model


def predict_model(seq_data, weights_file):
    """
    Prediction on trained model
    :param seq_data: test data
    :param weights_file: model weights file
    :return: prediction for seq_data
    """
    print("Prediction ...")
    model = load_model(weights_file)
    y_predict = model.predict(seq_data, verbose=1)
    return y_predict

