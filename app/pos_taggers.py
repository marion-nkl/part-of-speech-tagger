from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from nltk.corpus import treebank
from nltk.tag import CRFTagger
from nltk.tag import hmm
from sklearn.metrics import accuracy_score


class Tagger:
    def __init__(self, tagger_type):
        assert tagger_type in ['hmm', 'crf']

        self.X = None
        self.tagger = None
        self.tagger_type = tagger_type
        self.train_data = None

    def fit(self, data):
        """
        Fits a tagging model to object's data based on object's tagger name
        :return: a tagger object
        """
        tagger = None
        self.X = data

        if self.tagger_type == 'hmm':
            # Setup a trainer with default(None) values
            # And train with the data
            trainer = hmm.HiddenMarkovModelTrainer()
            tagger = trainer.train_supervised(data)

        elif self.tagger_type == 'crf':
            trainer = CRFTagger()
            trainer.train(self.train_data, 'model.crf.tagger')
            tagger = trainer

        self.tagger = tagger

        return tagger

    def evaluate(self, data):
        """
        Performs tagging in a dataset
        :param data: list of lists with tuples of words and POS tags
        :return: list with the actual labels of the input data, list with the predicted labels of the input data
        """
        y_pred = None
        sentences_of_tokens = list()
        y_true_tags = list()
        for sentence in data:
            words = [t[0] for t in sentence]
            tags = [t[1] for t in sentence]
            sentences_of_tokens.append(words)
            y_true_tags.append(tags)

        # Predicting pos tag labels for all sentences in our testing set
        if self.tagger:
            y_pred = [self.tagger.tag(xseq) for xseq in sentences_of_tokens]

        else:
            raise NotImplementedError('The Tagger is not Trained. Please perform training first.')

        # flattens the results for the list of lists of tuples
        y_true_flat = list(chain.from_iterable(y_true_tags))
        y_pred_flat = [t[1] for sublist in y_pred for t in sublist]

        accuracy = accuracy_score(y_true_flat, y_pred_flat)

        return {'accuracy': accuracy}

    def create_benchmark_plot(self,
                              train,
                              test,
                              n_splits=20,
                              plot_outfile=None,
                              y_ticks=0.025,
                              min_y_lim=0.0):
        """
        This method runs benchmarking for a crf model in order to check whether the classifier is learning.
        Also, learning curves are created.

        :param train: list. A list of lists of (word, pos-tag) tuples.
        :param test: list. A list of lists of (word, pos-tag) tuples.
        :param n_splits: int. Number of splits for the benchmarking. 20 splits every 5% of the training dataset.
        :param plot_outfile: str. A string in order to save the plot on disk.
        :param y_ticks: float. Number that defines the y_ticks.
        :param min_y_lim: float. Number that defines the minimum y limit of accuracy for the plot.
        """

        # placeholder for the metadata
        results = {'train_size': [], 'on_test': [], 'on_train': []}

        # calculating the batch size.
        split_size = int(len(train) / n_splits)

        # setting parameters for the graph.
        font_p = FontProperties()
        font_p.set_size('small')
        fig = plt.figure()
        fig.suptitle('Learning Curves', fontsize=20)
        ax = fig.add_subplot(111)
        ax.axis(xmin=0, xmax=len(train) * 1.05, ymin=0, ymax=1.1)
        plt.xlabel('N. of training instances', fontsize=18)
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
        batch_size = split_size

        for num in range(n_splits):
            # each time adds up (concatenates) a new batch.
            train_x_part = train[:batch_size]
            batch_size += split_size

            print(20 * '*')
            print('Split {} size: {}'.format(num, len(train_x_part)))

            results['train_size'].append(len(train_x_part))

            # fitting the model for the ginen sub training set
            self.fit(data=train_x_part)

            # checking the results always on the same test set
            result_on_test = self.evaluate(data=test)
            results['on_test'].append(result_on_test['accuracy'])

            # calculates the metrics for the given training part
            result_on_train_part = self.evaluate(data=train_x_part)
            results['on_train'].append(result_on_train_part['accuracy'])

            print('Train Acc: {}'.format(round(100 * result_on_train_part['accuracy']), 2))
            print('Test Acc: {}'.format(round(100 * result_on_test['accuracy'], 2)))

            line_up, = ax.plot(results['train_size'], results['on_train'], 'o-', label='Accuracy on Train')
            line_down, = ax.plot(results['train_size'], results['on_test'], 'o-', label='Accuracy on Test')

            plt.legend([line_up, line_down], ['Accuracy on Train', 'Accuracy on Test'], prop=font_p)

        if plot_outfile:
            fig.savefig(plot_outfile)

        plt.show()

        return results


if __name__ == '__main__':
    data = treebank.tagged_sents()[:3000]

    # fit HMM model
    train_set = data[:2000]
    test_set = data[-1000:]

    tagger_obj = Tagger('hmm')
    tagger_obj.create_benchmark_plot(train=train_set, test=test_set)
