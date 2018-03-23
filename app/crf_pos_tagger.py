from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from pprint import pprint
from app.data_fetcher import DataFetcher
from app import MODELS_DIR
import os
from collections import Counter

CRF_MODEL_FILEPATH = os.path.join(MODELS_DIR, 'crf_model_en.crfsuite')


def get_word_features(sent, position):
    """

    :param sent:
    :param position:
    :return:
    """

    # extracting the word
    word = sent[position][0]

    features = [
        'bias',
        'word.lower={}'.format(word.lower()),  # the word in lower
        'last_three_chars={}'.format(word[-3:]),  # up to last three characters
        'last_two_chars={}'.format(word[-2:]),  # up to last two characters
        'first_two_chars={}'.format(word[:2]),  # up to first two characters
        'first_three_chars={}'.format(word[:2]),  # up dto frist three characters
        'word.isupper={}'.format(word.isupper()),  # checks whether the word is written in uppercase.
        'word.istitle={}'.format(word.istitle()),  # checks if the first letter is in uppercase.
        'word.isdigit={}'.format(word.isdigit()),  # checks whether it is a digit.
        'word.endswith.ed={}'.format(word.endswith('ed')),
        'word.endswith.ing={}'.format(word.endswith('ing'))
    ]

    if position > 0:
        previous_word = sent[position - 1][0]
        features.extend([
            '-1:word.lower=' + previous_word.lower(),
            '-1:word.istitle={}'.format(previous_word.istitle()),
            '-1:word.isupper={}'.format(previous_word.isupper()),
            '-1:word.isdigit={}'.format(previous_word.isdigit()),
            '-1:word.endswith.ed={}'.format(previous_word.endswith('ed')),
            '-1:word.endswith.ing={}'.format(previous_word.endswith('ing'))

        ])
    else:
        features.append('BOS')

    if position < len(sent) - 1:
        next_word = sent[position + 1][0]
        features.extend([
            '+1:word.lower=' + next_word.lower(),
            '+1:word.istitle={}'.format(next_word.istitle()),
            '+1:word.isupper={}'.format(next_word.isupper()),
            '+1:word.isdigit={}'.format(next_word.isdigit()),
            '+1:word.endswith.ed={}'.format(next_word.endswith('ed')),
            '+1:word.endswith.ing={}'.format(next_word.endswith('ing'))
        ])
    else:
        features.append('EOS')

    return features
