from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from app.data_fetcher import DataFetcher
from app.evaluation import create_report

plt.style.use('ggplot')
pd.set_option('display.expand_frame_repr', False)


class Baseline:
    def __init__(self):
        self.model = None

    def fit(self, X):
        """
        Fits a baseline model that computes the most frequent POS tag for each word.
        :return: dict with keys the words of the vocabulary of the corpus and as values their POS tag
        """
        # pre processes the data. Splitting in (word, tag) tuples
        out = DataFetcher.pre_processing(X)
        # creating a dataframe with word and pos_tag columns.
        df = pd.DataFrame(out, columns=['word', 'pos_tag'])

        out_dict = dict()
        # grouping by the word in order to count the most common pos tag for each word
        for w, group in df.groupby(['word']):
            out_dict[w] = Counter(group['pos_tag']).most_common(1)[0][0]

        # calculates the most common tag for the whole corpus.
        out_dict['most_common_tag'] = df['pos_tag'].mode()[0]

        self.model = out_dict

        return self.model

    def predict(self, X):
        """
        Predicts the pos tags for a given list of words.
        :param X:
        :return:
        """

        def get_prediction(x):
            """

            :param x:
            :return:
            """
            if x:
                return self.model.get(x, self.model.get('most_common_tag'))

            return None

        return list(map(get_prediction, X))


def main(n_rarest_words=2500):
    """

    :param n_rarest_words: int. Fow how many of the rarest words, besides the whole evaluation, we should run the report
    :return:
    """
    # fetches and creates a dict containing the train, dev and test data.
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    # concatenates the train and dev sets and feed them to the baseline algorithm.
    baseline = Baseline()
    # fitting both the train and development data.
    baseline.fit(train_data + dev_data)

    # creating a dataframe with the test words and the true (actual) pos tags.
    test_tuples_df = pd.DataFrame(DataFetcher.pre_processing(test_data), columns=['word', 'y_true'])

    # predicting the pos tags in order to calculate the accurasy and the classification report
    test_tuples_df['y_pred'] = baseline.predict(test_tuples_df['word'])

    # extracting the classes of the tags in order to pass it for the classification report.
    classes = sorted(set(test_tuples_df['y_pred']) | set(test_tuples_df['y_pred']))

    # creating a report for the whole test dataset.
    create_report(test_tuples_df['y_true'], test_tuples_df['y_pred'], None)

    # calculating the number (ratio) of each distinct word in the test dataset.
    x = test_tuples_df.groupby(['word']).count() / len(test_tuples_df)

    # extracting the n rarest words by sorting
    top_n_rarest_words = x.sort_values('y_true').reset_index()['word'][:n_rarest_words]

    # extracting the dataframe of the n rarest words in order to check if
    # we have the same accuracy as in the whole dataset
    rarest_words_df = test_tuples_df[test_tuples_df['word'].isin(top_n_rarest_words)].reset_index(drop=True)

    print(len(x))
    print('--------------------------------------------------------------------------------')
    print('{} rarest words classification report results: '.format(n_rarest_words), end='\n\n')
    print('--------------------------------------------------------------------------------')
    create_report(rarest_words_df['y_true'], rarest_words_df['y_pred'], None)

    # sorting the words by their frequency in order to create a plot
    to_plot = x.sort_values('y_true', ascending=False)['y_true'] * 100
    to_plot.reset_index(drop=True, inplace=True)
    to_plot.plot(title='Words Frequency from Most Common to Rarest', grid=True)
    plt.ylabel('Percentage of Occurances')
    plt.show()

    # creating a bar plot for each pos tag by calculating the occurrences for each one tag.
    tags_freqs = test_tuples_df.groupby('y_true').count()['word'] / len(test_tuples_df)
    tags_freqs *= 100
    tags_freqs.plot(kind='bar', title='Percentage of Tags on Test set')
    plt.ylabel('Percentage')
    plt.xlabel('Pos Tags')
    plt.show()


if __name__ == '__main__':
    main()
