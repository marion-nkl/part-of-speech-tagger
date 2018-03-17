from app.data_fetcher import DataFetcher
import pandas as pd
from collections import Counter


class Baseline:
    def __init__(self, data):
        self.data = data
        self.model = None

    def fit(self):
        """
        Fits a baseline model that computes the most frequent POS tag for each word.
        :return: dict with keys the words of the vocabulary of the corpus and as values their POS tag
        """
        out = DataFetcher.pre_processing(self.data)
        df = pd.DataFrame(out, columns=['word', 'pos_tag'])

        out_dict = dict()
        for w, group in df.groupby(['word']):
            out_dict[w] = Counter(group['pos_tag']).most_common(1)[0][0]

        self.model = out_dict

if __name__ == '__main__':

    data_dict = DataFetcher.read_data(files_list=['train', 'dev'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])

    # concatenates the train and dev sets and feed them to the baseline algo
    baseline = Baseline(train_data + dev_data)
    baseline.fit()
    print(baseline.model)
