from nltk.corpus import treebank
from nltk.tag import CRFTagger
from nltk.tag import hmm

from app.data_fetcher import DataFetcher


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

    data = treebank.tagged_sents()[:3000]

    data_dict = DataFetcher.read_data()
    train_data = DataFetcher.parse_conllu(data_dict['train'], 'xpostag')
    cleaned_train_data = DataFetcher.remove_empty_sentences(train_data)

    dev = [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]
    test = 'This is a sentence'.split()

    hmm_obj = Tagger(cleaned_train_data, 'hmm')
    hmm_obj.fit()

    hmm_obj_nltk = Tagger(data, 'hmm')
    hmm_obj_nltk.fit()

    print('-' * 30)
    print('HMM Tagger')
    print('Tagging with our dataset: {}'.format(hmm_obj.tagger.tag(test)))
    print('Evaluation: {}'.format(hmm_obj.tagger.evaluate([dev])))
    print('Tagging with nltk corpus: {}'.format(hmm_obj_nltk.tagger.tag(test)))
    print('Evaluation: {}'.format(hmm_obj_nltk.tagger.evaluate([dev])))

    print()
    print()

    # crf_obj = Tagger(cleaned_train_data, 'crf')
    # crf_obj.fit()
    #
    # crf_obj_nltk = Tagger(data, 'crf')
    # crf_obj_nltk.fit()
    #
    # print('-' * 30)
    # print('CRF Tagger')
    # print('Tagging with our dataset: {}'.format(crf_obj.tagger.tag_sents([test])))
    # print('Evaluation: {}'.format(crf_obj.tagger.evaluate([dev])))
    # print('Tagging with nltk corpus: {}'.format(crf_obj_nltk.tagger.tag_sents([test])))
    # print('Evaluation: {}'.format(crf_obj_nltk.tagger.evaluate([dev])))
    # print()
