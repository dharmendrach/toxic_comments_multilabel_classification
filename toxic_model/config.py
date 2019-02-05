from os.path import join, dirname

"""
Global config options
"""

# base directory of project
BASE_DIR = dirname(dirname(__file__))

# train, test and sample submission file path
DATA_DIR = join(BASE_DIR, "data")
TRAIN_CSV = join(DATA_DIR, "train.csv")
TEST_CSV = join(DATA_DIR, "test.csv")
SUBMISSION_FILE = join(DATA_DIR, "sample_submission.csv")

APOSTROPHE_FILE = join(DATA_DIR, "apostrophe.json")  # apostrophe json file

# embedding files
FASTTEXT_WIKI = join(DATA_DIR, 'wiki.en.vec')
FASTTEXT_CRAWL = join(DATA_DIR, 'crawl-300d-2M.vec')
GLOVE_840B = join(DATA_DIR, "glove.840B.300d.txt")
GLOVE_TWITTER = join(DATA_DIR, "glove.twitter.27B.200d.txt")

# saved data directory, weights file, config files and predictions
OUTPUT_DIR = join(BASE_DIR, "output", "{}")
CONFIG_JSON = join(OUTPUT_DIR, "config.json")
DATA_PKL = join(OUTPUT_DIR, "data.pkl")
WEIGHT_FILE = join(OUTPUT_DIR, "weights_{}.best.hdf5")
TEST_SUBMISSION_FILE = join(OUTPUT_DIR, "output.csv")  # saved test prediction
