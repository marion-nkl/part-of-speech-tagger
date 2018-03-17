from app.data_fetcher import DataFetcher
import pandas as pd
from collections import Counter
from app.evaluation import create_report

pd.set_option('display.expand_frame_repr', False)


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

        out_dict['most_common_tag'] = df['pos_tag'].mode()[0]

        self.model = out_dict


def main():
    """

    :return:
    """
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    # concatenates the train and dev sets and feed them to the baseline algo
    baseline = Baseline(train_data + dev_data)
    baseline.fit()

    test_tuples_df = pd.DataFrame(DataFetcher.pre_processing(test_data), columns=['word', 'y_true'])
    test_tuples_df['y_pred'] = test_tuples_df['word'].map(baseline.model)
    test_tuples_df['y_pred'].fillna(value=baseline.model.get('most_common_tag'), inplace=True)

    classes = sorted(set(test_tuples_df['y_pred']) | set(test_tuples_df['y_pred']))

    create_report(y_true=test_tuples_df['y_true'],
                  y_pred=test_tuples_df['y_pred'],
                  classes=classes)


if __name__ == '__main__':
    main()
