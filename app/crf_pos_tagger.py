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