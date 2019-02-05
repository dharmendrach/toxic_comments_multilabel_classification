import re
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing import text, sequence

from toxic_model.config import APOSTROPHE_FILE, FASTTEXT_WIKI, FASTTEXT_CRAWL, GLOVE_840B, GLOVE_TWITTER
from toxic_model.utils import load_json

APPO = load_json(APOSTROPHE_FILE)
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemat = WordNetLemmatizer()
preprocess_tokenizer = TweetTokenizer()


def process_comment(comment, preprocess_config):
    """
    Converting to lowercase, removing stopwords, lemmatization, stemming etc.
    :param comment: comment text
    :param preprocess_config: configurations for preprocessing comments
    :return: processed comment text
    """
    comment = comment.lower()  # converting to lowercase

    # removing end of lines and ip address
    comment = re.sub("\\n", "", comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)

    # Tokenize using Tweet tokenizer
    comment = preprocess_tokenizer.tokenize(comment)

    # Converting Apostrophe
    if preprocess_config['remove_apostrophe']:
        comment = [APPO[word] if word in APPO else word for word in comment]

    # removing stopwords if config enabled
    if preprocess_config['remove_stopwords']:
        comment = [word for word in comment if word not in stopwords]

    # Converting to stem words
    if preprocess_config['stemming']:
        comment = [stemmer.stem(word) for word in comment]

    # lemmatization verb level
    if preprocess_config['lemmatization']:
        comment = [lemat.lemmatize(word, "v") for word in comment]

    comment = " ".join(comment)

    # Regex for only keeping alpha_numeric characters
    if preprocess_config['alpha_numeric']:
        comment = re.sub("\W+", " ", comment)

    return comment


def prepare_data(train_file, test_file, preprocess_config):
    """
    Read train and test files and convert to sequence data
    :param train_file: train csv file
    :param test_file: test csv file
    :param preprocess_config: configurations for processing data
    :return: data dictionary containing train and test sequence data, train labels, word_index
    """

    data_dict = {}

    # reading training file
    print("processing training file ...")
    train_data = pd.read_csv(train_file)
    train_data['comment_text'].fillna(value="NAN", inplace=True)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    data_dict['train_labels'] = train_data[list_classes].values
    # processing train comments
    train_comments = train_data['comment_text'].apply(process_comment, args=(preprocess_config,))

    # reading testing file
    print("processing testing file ...")
    test_data = pd.read_csv(test_file)
    test_data['comment_text'].fillna(value="NAN", inplace=True)
    test_comments = test_data['comment_text'].apply(process_comment, args=(preprocess_config,))

    # fitting tokenizer on train and test data
    print("fitting tokenizer ...")
    tokenizer = text.Tokenizer(num_words=preprocess_config['max_nb_words'])
    tokenizer.fit_on_texts(list(train_comments.values) + list(test_comments.values))

    # adding word index to data_dict for embedding matrix creation
    data_dict['word_index'] = tokenizer.word_index
    print("number of unique words :", len(tokenizer.word_index))

    # converting train data to sequences
    print("tokenize training data...")
    train_seq = tokenizer.texts_to_sequences(train_comments)
    data_dict['train_seq'] = sequence.pad_sequences(train_seq, maxlen=preprocess_config['max_seq_len'])

    # converting test data to sequences
    print("tokenize test data...")
    test_seq = tokenizer.texts_to_sequences(test_comments)
    data_dict['test_seq'] = sequence.pad_sequences(test_seq, maxlen=preprocess_config['max_seq_len'])

    return data_dict


def get_embedding_matrix(embedding_type, embedding_size, word_dict):
    """
    Read fasttext and glove embedding files and create embedding matrix for given word to index dictionary
    :param embedding_type: fasttext or glove
    :param embedding_size: embedding dimensions
    :param word_dict: word to index dictionary for creating embedding_matrix
    :return: embedding matrix for given word dict
    """

    # only support fasttext wiki, crawl, glove 840B and glove twitter embedding
    supported_embedding = ['glove_840B', 'glove_twitter', 'fasttext_wiki', 'fasttext_crawl']
    embedding_files = [GLOVE_840B, GLOVE_TWITTER, FASTTEXT_WIKI, FASTTEXT_CRAWL]
    assert embedding_type in supported_embedding

    # will arrange word vectors according to word_to_index dictionary
    embedding_matrix = np.zeros((len(word_dict) + 1, embedding_size), dtype='float32')

    filename = embedding_files[supported_embedding.index(embedding_type)]
    embed_file = open(filename, encoding='utf-8')

    # removing first line in case of fasttext embedding
    if embedding_type in ['fasttext_wiki', 'fasttext_crawl']:
        start_line = embed_file.readline()
        print("fasttext vocab size, embedding size :", start_line)

    found_words = 0
    for line in tqdm(embed_file, file=sys.stdout):
        values = line.rstrip().split(' ')
        # adding word vector to matrix for words in train and test data
        if values[0] in word_dict:
            embedding_matrix[word_dict[values[0]]] = np.asarray(values[1:], dtype='float32')
            found_words += 1

    print("Number of unique words:", len(word_dict))
    print("Number of missed words:", len(word_dict)-found_words)
    return embedding_matrix
