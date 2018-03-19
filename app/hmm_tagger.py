# Import the toolkit and tags
import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from app.data_fetcher import DataFetcher


class HMMTager:
    def __init__(self, data):
        self.train_data = data  # list
        self.tagger = None

    def fit(self):
        # Setup a trainer with default(None) values
        # And train with the data
        trainer = hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train_supervised(self.train_data)

        self.tagger = tagger

if __name__ == '__main__':

    data = treebank.tagged_sents()[:3000]

    tag_obj = HMMTager(data)
    tag_obj.fit()

    data_d = DataFetcher.read_data()

    train_data = DataFetcher.parse_conllu(data_d['train'])

    print(train_data[:30])

    print(tag_obj.tagger.tag('This is a sentence'.split()))
