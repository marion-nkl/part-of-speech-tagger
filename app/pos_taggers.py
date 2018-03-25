from nltk.corpus import treebank
from nltk.tag import CRFTagger
from nltk.tag import hmm
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from app.data_fetcher import DataFetcher
from app.evaluation import tagger_classification_report


class Tagger:
    def __init__(self, data, tagger_name):
        self.train_data = data  # list
        self.tagger = None
        self.tagger_name = tagger_name

    def fit(self):
        """
        Fits a tagging model to object's data based on object's tagger name
        :return: a tagger object
        """
        tagger = None
        if self.tagger_name == 'hmm':
            # Setup a trainer with default(None) values
            # And train with the data
            trainer = hmm.HiddenMarkovModelTrainer()
            tagger = trainer.train_supervised(self.train_data)

        elif self.tagger_name == 'crf':
            trainer = CRFTagger()
            trainer.train(self.train_data, 'model.crf.tagger')
            tagger = trainer

        self.tagger = tagger


if __name__ == '__main__':

    def extract_tokens_from_sentence_token_tuples(sent):
        """
        This function extracts the str tokens from an iterable of (token, labels) tuples.

        :param sent: list. An iterable of (word, pos_tag) tuples.
        :return: list. A lists of strings.
        """
        return [token for token, pos_tag in sent]

    def extract_pos_tags_from_sentence_token_tuples(sent):
        """
        This function extracts the pos tags (labels in general) from a given iterable of (token, label) tuples.

        :param sent: list. An iterable of (word, pos-tag) tuples.
        :return: list. An iterable of pos tags.
        """
        return [postag for token, postag in sent]

    data = treebank.tagged_sents()[:3000]

    # data_dict = DataFetcher.read_data()
    # train_data = DataFetcher.parse_conllu(data_dict['train'], 'xpostag')
    # cleaned_train_data = DataFetcher.remove_empty_sentences(train_data)
    #
    # dev = [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]
    # test = 'This is a sentence'.split()
    #
    # hmm_obj = Tagger(cleaned_train_data, 'hmm')
    # hmm_obj.fit()

    # train data
    # data_dict = DataFetcher.read_data()
    # train_data = DataFetcher.parse_conllu(data_dict['train'])
    # dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    # cleaned_train_data = DataFetcher.remove_empty_sentences(train_data + dev_data)

    # print('-' * 30)
    # print('HMM Tagger')
    # print('Tagging with our dataset: {}'.format(hmm_obj.tagger.tag(test)))
    # print('Evaluation: {}'.format(hmm_obj.tagger.evaluate([dev])))
    # print('Tagging with nltk corpus: {}'.format(hmm_obj_nltk.tagger.tag(test)))
    # print('Evaluation: {}'.format(hmm_obj_nltk.tagger.evaluate([dev])))


    # fit HMM model
    tagger = Tagger(data[:2000], 'hmm')
    tagger.fit()

    # Tag sentences
    results = list()
    true_values = list()
    for s in data[-1000:]:
        true_values.append(extract_pos_tags_from_sentence_token_tuples(s))
        results.append(extract_pos_tags_from_sentence_token_tuples(tagger.tagger.tag(extract_tokens_from_sentence_token_tuples(s))))

    print(results[:3])
    print(true_values[:3])

    # evaluation report
    print(tagger_classification_report(true_values, results)['clf_report'])



