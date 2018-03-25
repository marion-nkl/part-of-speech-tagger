from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.metrics import f1_score, accuracy_score

from app.data_fetcher import DataFetcher
from app.evaluation import create_report

plt.style.use('ggplot')
pd.set_option('display.expand_frame_repr', False)
plt.rcParams['figure.figsize'] = (16, 8)


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
    create_report(test_tuples_df['y_true'], test_tuples_df['y_pred'], classes)

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
    plt.ylabel('Percentage of Occurrences')
    plt.show()

    # creating a bar plot for each pos tag by calculating the occurrences for each one tag.
    tags_freqs = test_tuples_df.groupby('y_true').count()['word'] / len(test_tuples_df)
    tags_freqs *= 100
    tags_freqs.plot(kind='bar', title='Percentage of Tags on Test set')
    plt.ylabel('Percentage')
    plt.xlabel('Pos Tags')
    plt.show()


def baseline_benchmark(train, test):
    """
    This function calculates metrics for evaluation of a Baseline POS TAGGER classifier over a training and a test set.

    :param train: list
    :param test: list
    :return: dict
    """
    baseline = Baseline()
    # fitting both the train data
    baseline.fit(X=train)

    # creating a dataframe with the test words and the true (actual) pos tags.
    test_tuples_df = pd.DataFrame(DataFetcher.pre_processing(test), columns=['word', 'y_true'])

    # predicting the pos tags in order to calculate the accuracy and the classification report
    test_tuples_df['y_pred'] = baseline.predict(test_tuples_df['word'])
    test_tuples_df['y_pred'].fillna(baseline.model['most_common_tag'], inplace=True)

    y_true = test_tuples_df['y_true']
    y_pred = test_tuples_df['y_pred']

    if y_pred.isnull().sum() > 0:
        print(test_tuples_df[test_tuples_df['y_pred'].isnull()])

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {'accuracy': accuracy, 'f1': f1}


def create_baseline_benchmark_plot(train,
                                   test,
                                   n_splits=20,
                                   plot_outfile=None,
                                   y_ticks=0.025,
                                   min_y_lim=0.4):
    """

    :param train:
    :param test:
    :param n_splits:
    :param plot_outfile:
    :param y_ticks:
    :param min_y_lim:
    :return:
    """

    results = {'train_size': [], 'on_test': [], 'on_train': []}

    split_size = int(len(train) / n_splits)

    # setting parameters for the graph.
    font_p = FontProperties()

    font_p.set_size('small')

    fig = plt.figure()
    fig.suptitle('Baseline POS Tagger Learning Curves', fontsize=16)

    ax = fig.add_subplot(111)
    ax.axis(xmin=0, xmax=len(train) * 1.05, ymin=0, ymax=1.1)

    plt.xlabel('N. of training instances', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.grid(True)

    plt.axvline(x=int(len(train) * 0.3))
    plt.yticks(np.arange(0, 1.025, 0.025))

    if y_ticks == 0.05:
        plt.yticks(np.arange(0, 1.025, 0.05))

    elif y_ticks == 0.025:
        plt.yticks(np.arange(0, 1.025, 0.025))

    plt.ylim([min_y_lim, 1.025])

    # each time adds up one split and refits the model.
    counter = split_size
    for i in range(n_splits):
        train_x_part = train[:counter]
        counter += split_size

        print(20 * '*')
        print('Split {} size: {}'.format(i, len(train_x_part)))

        results['train_size'].append(len(train_x_part))

        result_on_test = baseline_benchmark(train=train_x_part, test=test)

        # calculates each time the metrics also on the test.
        results['on_test'].append(result_on_test['accuracy'])

        # calculates the metrics for the given training part
        result_on_train_part = baseline_benchmark(train=train_x_part, test=train_x_part)

        results['on_train'].append(result_on_train_part['accuracy'])

        line_up, = ax.plot(results['train_size'], results['on_train'], 'o-', label='Accuracy on Train')

        line_down, = ax.plot(results['train_size'], results['on_test'], 'o-', label='Accuracy on Test')

        plt.legend([line_up, line_down], ['Accuracy on Train', 'Accuracy on Test'], prop=font_p)

    if plot_outfile:
        fig.savefig(plot_outfile)

    plt.show()

    return results


if __name__ == '__main__':
    # main()

    # fetches and creates a dict containing the train, dev and test data.
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    train_dev_data = train_data + dev_data

    create_baseline_benchmark_plot(train=train_dev_data, test=test_data)
