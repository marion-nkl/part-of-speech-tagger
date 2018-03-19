from app.data_fetcher import DataFetcher
import pandas as pd
from collections import Counter
from app.evaluation import create_report
import matplotlib.pyplot as plt

plt.style.use('ggplot')
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


def main(n_rarest_words=2500):
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

    create_report(test_tuples_df['y_true'], test_tuples_df['y_pred'], classes)

    x = test_tuples_df.groupby(['word']).count() / len(test_tuples_df)
    top_n_rarest_words = x.sort_values('y_true').reset_index()['word'][:n_rarest_words]

    rarest_words_df = test_tuples_df[test_tuples_df['word'].isin(top_n_rarest_words)].reset_index(drop=True)
    print(len(x))
    print('--------------------------------------------------------------------------------')
    print('{} most rare words results: '.format(n_rarest_words), end='\n\n')
    print('--------------------------------------------------------------------------------')
    create_report(rarest_words_df['y_true'], rarest_words_df['y_pred'], None)

    to_plot = x.sort_values('y_true', ascending=False)['y_true'] * 100
    to_plot.reset_index(drop=True, inplace=True)
    to_plot.plot(title='Words Frequency from Most Common to Rarest', grid=True)
    plt.ylabel('Percentage of Occurances')
    plt.show()

    tags_freqs = test_tuples_df.groupby('y_true').count()['word'] / len(test_tuples_df)
    tags_freqs *= 100
    tags_freqs.plot(kind='bar', title='Percentage of Tags on Test set')
    plt.ylabel('Percentage')
    plt.xlabel('Pos Tags')
    plt.show()


if __name__ == '__main__':
    main()
