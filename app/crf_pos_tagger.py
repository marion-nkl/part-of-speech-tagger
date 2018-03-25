import itertools as it
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycrfsuite
from matplotlib.font_manager import FontProperties
from numpy import mean
from numpy import random

from app import MODELS_DIR
from app.data_fetcher import DataFetcher
from app.evaluation import tagger_classification_report, print_crf_transitions

plt.rcParams['figure.figsize'] = (16, 8)
pd.set_option('display.expand_frame_repr', False)

CRF_MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'crf_model_en.crfsuite')
GRID_SEARCH_CRF_MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'temp_crf_model_en.crfsuite')


class CRFTagger:

    def __init__(self):
        """

        """
        self.tagger = None

    @staticmethod
    def feed_crf_params(grid_search_params=None):
        """
        This function provides all the combinations of a greedy search parameters.

        :param grid_search_params:
        :return: dict. A dictionary of parameters for the crf model.
        """
        if grid_search_params is None:
            grid_search_params = {
                'c1': [1.0, 0.1, 0.01],  # coefficient for L1 penalty
                'c2': [1e-3, 1e-2, 1e-1],  # coefficient for L2 penalty
                'max_iterations': [50, 100, 200],  # stop earlier
                'feature.possible_transitions': [True, False]  # include transitions that are possible, but not observed
            }

        # sorting the keys
        sorted_keys = sorted(grid_search_params)
        # creating all the possible combinations for all the parameters
        combinations = it.product(*(grid_search_params[key] for key in sorted_keys))

        # for each combination get the params in a dict
        for i in combinations:
            params = dict(zip(sorted_keys, i))
            yield params

    @staticmethod
    def feed_cross_validation(sentences, seed=1234, k_folds=5, verbose=0):
        """
        This method feeds the train and held_out data-sets in each iteration.
        :param sentences: list. An iterable of sentences
        :param seed: Int. A number that helps in the reproduction of the shuffling
        :param k_folds: Int. Number of folds.
        :param verbose: Int. Defines the level of verbosity.
        :return: yields a dict of train and held_out data-sets.
        """

        # setting the seed in order to be able to reproduce results.
        np.random.seed(seed)

        # shuffling the list of sentences.
        if verbose > 0: print('Shuffling data set in order to brake to train and held_out data-sets')
        random.shuffle(sentences)

        # calculating the ratios in actual numbers
        total_len = len(sentences)

        # split_size
        split_size = int(total_len / float(k_folds))

        if verbose > 0:
            print('Splitting data-set in {} folds'.format(k_folds))
            print('Split size for held out dataset: {}'.format(split_size))
            print('Split size for training dataset: {}'.format(total_len - split_size))

        for i in range(1, k_folds + 1):
            yield {'held_out': sentences[(i - 1) * split_size: i * split_size],
                   'train': sentences[:(i - 1) * split_size] + sentences[i * split_size:]}

    @staticmethod
    def get_word_features(sent, position):
        """

        :param sent: list. A list of (word, pos_tag) tuples
        :param position: int. A integer defining the position (state) of the word
        :return: list. A list of features created for the given word
        """

        # extracting the word
        word = sent[position][0]

        features = [
            'bias',
            'word.lower={}'.format(word.lower()),  # the word in lower
            'last_three_chars={}'.format(word[-3:]),  # up to last three characters
            'last_two_chars={}'.format(word[-2:]),  # up to last two characters
            'first_two_chars={}'.format(word[:2]),  # up to first two characters
            'first_three_chars={}'.format(word[:2]),  # up dto first three characters
            'word.isupper={}'.format(word.isupper()),  # checks whether the word is written in uppercase.
            'word.istitle={}'.format(word.istitle()),  # checks if the first letter is in uppercase.
            'word.isdigit={}'.format(word.isdigit()),  # checks whether it is a digit.
            'word.endswith.ed={}'.format(word.endswith('ed')),
            'word.endswith.ing={}'.format(word.endswith('ing'))
        ]

        # previous state's features
        if position > 0:
            previous_word = sent[position - 1][0]
            features.extend([
                '-1:word.lower=' + previous_word.lower(),
                '-1:word.istitle={}'.format(previous_word.istitle()),
                '-1:word.isupper={}'.format(previous_word.isupper()),
                '-1:word.isdigit={}'.format(previous_word.isdigit()),
                '-1:word.endswith.ed={}'.format(previous_word.endswith('ed')),
                '-1:word.endswith.ing={}'.format(previous_word.endswith('ing'))

            ])
        else:
            features.append('BOS')

        # next state's features
        if position < len(sent) - 1:
            next_word = sent[position + 1][0]
            features.extend([
                '+1:word.lower=' + next_word.lower(),
                '+1:word.istitle={}'.format(next_word.istitle()),
                '+1:word.isupper={}'.format(next_word.isupper()),
                '+1:word.isdigit={}'.format(next_word.isdigit()),
                '+1:word.endswith.ed={}'.format(next_word.endswith('ed')),
                '+1:word.endswith.ing={}'.format(next_word.endswith('ing'))
            ])
        else:
            features.append('EOS')

        return features

    def get_sentence_to_features(self, sent):
        """
        This function creates features for all tokens of a sentence.

        :param sent: A list of (token, pos tag) tuples.
        :return: list. An iterable or lists containing features for each token.
        """

        return [self.get_word_features(sent, num) for num in range(len(sent))]

    def print_crf_tagger_example(self, sentence):
        """
        This function runs an POS TAGGING example for a given sentence
        :param sentence: List. An iterable of (token, pos tag) tuples.
        :return: None.
        """

        if self.tagger is not None:
            print('\n\nSentence: "{}"'.format(' '.join(self.extract_tokens_from_sentence_token_tuples(sentence)),
                                              end='\n\n'))

            print("Predicted:", ' '.join(self.tagger.tag(self.get_sentence_to_features(sentence))))
            print("Correct:  ", ' '.join(self.extract_pos_tags_from_sentence_token_tuples(sentence)))

    @staticmethod
    def extract_pos_tags_from_sentence_token_tuples(sent):
        """
        This function extracts the pos tags (labels in general) from a given iterable of (token, label) tuples.

        :param sent: list. An iterable of (word, pos-tag) tuples.
        :return: list. An iterable of pos tags.
        """
        return [postag for token, postag in sent]

    @staticmethod
    def extract_tokens_from_sentence_token_tuples(sent):
        """
        This function extracts the str tokens from an iterable of (token, labels) tuples.

        :param sent: list. An iterable of (word, pos_tag) tuples.
        :return: list. A lists of strings.
        """
        return [token for token, pos_tag in sent]

    def fit(self, X, params=None, filepath='crf_model_en.crfsuite'):
        """
        This model trains CRF model for POS TAGS.

        :param X: List. A list of lists of (word, pos_tags) tuples.
        :param params: dict. A dictionary containing the hyper parameters for the crf model.
        :param filepath: str. A file name in order to save the model.
        :return: obj. A tagger object.
        """
        filepath = os.path.join(MODELS_DIR, filepath)

        if params is None:
            # instantiating parameters in the case of not providing parameters.
            # We use L-BFGS training algorithm (it is default) + Elastic Net (L1+L2) regularization
            params = {
                'c1': 1.0,  # coefficient for L1 penalty
                'c2': 1e-3,  # coefficient for L2 penalty
                'max_iterations': 50,  # stop earlier
                'feature.possible_transitions': True}  # include transitions that are possible, but not observed

        # extracting the features and the labels (pos tags) for the train examples
        x_features = [self.get_sentence_to_features(s) for s in X]
        y_pos_tags = [self.extract_pos_tags_from_sentence_token_tuples(s) for s in X]

        # Training the model: In order to train the model, we create a pycrfsuite.Trainer, load the training data and
        # calling the 'train' method
        trainer = pycrfsuite.Trainer(verbose=False)
        for x_seq, y_seq in zip(x_features, y_pos_tags):
            trainer.append(x_seq, y_seq)

        # setting the parameters for the models.
        trainer.set_params(params)
        # training the actual model and saving it to the models directory
        trainer.train(filepath)

        # Make predictions
        # In order to use the trained model, we create a pycrfsuite.Tagger, open the model and use "tag" method:
        tagger = pycrfsuite.Tagger()
        tagger.open(filepath)

        self.tagger = tagger

        return tagger

    def predict(self, X):
        """
        This method uses a pre trained tagger in order to predict new pos tags for a given set of sentences.
        :param X: list. A list of lists of (word, pos-tag) tuples
        :return: list. A list of the predicted pos tags.
        """
        y_pred = None

        # extracting the features and the labels (pos tags) for the test examples
        X_test = [self.get_sentence_to_features(s) for s in X]

        # Predicting pos tag labels for all sentences in our testing set
        if self.tagger:
            y_pred = [self.tagger.tag(xseq) for xseq in X_test]

        return y_pred

    def evaluate(self, X, verbose=0):
        """
        This method uses a pre trained crf pos tagger in order to make evaluations on known data sets.
        :param X: A list of lists of (word, pos-tag) tuples.
        :param verbose: Int. Level of verbosity
        :return: dict. A dictionary containing several metadata about the model's evaluation.
        """

        # extracting the labels (pos tags) for the sentences
        y_test = [self.extract_pos_tags_from_sentence_token_tuples(s) for s in X]

        # Predicting pos tag labels for all sentences in our set
        y_pred = self.predict(X)

        # calculating metrics and creating classification report.
        model_metadata = tagger_classification_report(y_test, y_pred)

        model_accuracy = model_metadata['accuracy']
        model_clf_report = model_metadata['clf_report']
        # Checking what classifier has learned
        info = self.tagger.info()

        if verbose > 0:
            print('Model Accuracy: {}'.format(model_accuracy), end='\n\n')

        if verbose > 1:
            print(model_clf_report)

            print("\nTop likely Pos Tags transitions:")
            # calculating the most likely transitions for POS TAGS sequences
            print_crf_transitions(Counter(info.transitions).most_common(10))

            print("\nTop unlikely Pos Tags transitions:")
            # calculating the least likely transitions for POS TAGS sequences
            print_crf_transitions(Counter(info.transitions).most_common()[-10:])

        model_metadata['model'] = self.tagger

        return model_metadata

    def grid_search_cross_validation(self,
                                     X_train,
                                     X_test,
                                     filepath='grid_crf_model_en.crfsuite',
                                     grid_params=None,
                                     verbose=0,
                                     k_folds=3):
        """
        This method runs k fold cross validation along with greedy search. For all combinations of parameters
        and for all folds, the function fits models, calculates accuracy scores, and average accuracy score for
        all folds. Then, the best model is selected in terms of accuracy in order train the model in the whole
        training dataset.

        :param X_train: List. An list of lists of tuples.
        :param X_test: List. An list of lists of tuples.
        :param grid_params: Dict. A dictionary containing parameters as keys and list of values as values.
        :param verbose: Int. Verbosity level.
        :param k_folds: Int. Number of splits for the k-fold cross validation
        :param filepath: Str. The filepath for the trained models to be stored.
        :return: dict. A dictionary containing the actual model, along with score metadata.
        """
        # obtaining an iterators of all the combinations of parameters
        param_combinations = self.feed_crf_params(grid_search_params=grid_params)

        # instantiating a list in order to store all the metadta for the gridsearch CV
        grid_search_results = list()

        for param_combination in param_combinations:
            # instantiate a dictionary that will contain all meetadata for this given set of parameters
            # and for all splits.
            d = dict()
            d['parameters'] = param_combination

            for k, v in param_combination.items():
                d['param-{}'.format(k)] = v

            scores = list()  # we need it in order to calculate the average score.

            if verbose > 0:
                print('Model Parameters: {}'.format(param_combination))

            for num, data in enumerate(self.feed_cross_validation(sentences=X_train,
                                                                  k_folds=k_folds)):
                # iterating through each different train dataset and obtaining different train and held out parts
                # of the original training dataset.
                train = data['train']
                held_out = data['held_out']

                verb = 1 if verbose > 1 else 0
                # fitting the model for the given train and held out parts
                self.fit(X=train, params=param_combination, filepath=filepath)

                model_meta = self.evaluate(X=held_out, verbose=verb)

                # extracting the accuracy score for this particular run. Storing it in order to have
                # all the metadata in one place.
                d["fold_{}_score".format(num + 1)] = model_meta['accuracy']
                scores.append(model_meta['accuracy'])

            d['mean_test_score'] = mean(scores)

            grid_search_results.append(d)

        # reverse sorting in order to get have the best results first.
        best_model_metadata = sorted(grid_search_results, key=lambda x: x['mean_test_score'], reverse=True)

        if verbose == 1:
            print(pd.DataFrame(best_model_metadata))

        # selecting the best parameters in order to retrain the model to the whole training dataset.
        best_parameters = best_model_metadata[0]['parameters']

        if verbose > 0:
            print('Best GridSearchCV CRF POS tagger params: {}'.format(best_parameters), end='\n\n')

        # retraining the model with the best parameters on the whole training dataset.
        self.fit(X=X_train, params=best_parameters, filepath=filepath)

        # evaluating the performance of the best model on the test dataset.
        best_model_meta = self.evaluate(X=X_test, verbose=verbose)

        # returning the performance metrics and metadata for the best model.
        return best_model_meta

    def create_benchmark_plot(self,
                              train,
                              test,
                              n_splits=20,
                              params=None,
                              plot_outfile=None,
                              y_ticks=0.025,
                              min_y_lim=0.4):
        """
        Thsi method runs benchmarking for a crf model in order to check whether the classifier is learing.
        Also, learning curves are created.

        :param train: list. A list of lists of (word, pos-tag) tuples.
        :param test: list. A list of lists of (word, pos-tag) tuples.
        :param n_splits: int. Number of splits for the benchmarking. 20 splits every 5% of the training dataset.
        :param params: dict. A dictionary containing the hyper parameters for the crf model.
        :param plot_outfile: str. A string in order to save the plot on disk.
        :param y_ticks: float. Number that defines the y_ticks.
        :param min_y_lim: float. Number that defines the minimum y limit of accuracy for the plot.
        :return:
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
            self.fit(X=train_x_part, params=params)

            # checking the results always on the same test set
            result_on_test = self.evaluate(X=test, verbose=1)
            results['on_test'].append(result_on_test['accuracy'])

            # calculates the metrics for the given training part
            result_on_train_part = self.evaluate(X=train_x_part, verbose=1)
            results['on_train'].append(result_on_train_part['accuracy'])

            line_up, = ax.plot(results['train_size'], results['on_train'], 'o-', label='Accuracy on Train')
            line_down, = ax.plot(results['train_size'], results['on_test'], 'o-', label='Accuracy on Test')

            plt.legend([line_up, line_down], ['Accuracy on Train', 'Accuracy on Test'], prop=font_p)

        if plot_outfile:
            fig.savefig(plot_outfile)

        plt.show()

        return results


def main_grid_search():
    """

    :return:
    """

    # fetches and creates a dict containing the train, dev and test data.
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    # extracting train, dev, and test datasets
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    # removing any empty sentences from the datasets.
    train_sentences = DataFetcher.remove_empty_sentences(train_data)
    dev_sentences = DataFetcher.remove_empty_sentences(dev_data)
    test_sentences = DataFetcher.remove_empty_sentences(test_data)

    # concatenating the train and held out (dev) dataset in order to feed it for cross validation.
    train_dev_sentences = train_sentences + dev_sentences

    # setting the Grid Search CV parameters. All combinations will be tested.
    grid_params = {
        'c1': [1.0, 0.1, 0.01],  # coeff for L1 penalty
        'c2': [1.0, 0.1, 0.01],  # coeff for L2 penalty
        'max_iterations': [50, 100, 200, 250],  # stop earlier
        'feature.possible_transitions': [True, False]}  # include transitions that are possible, but not observed

    crf_obj = CRFTagger()
    # Running grid search CV. Obtaining the best model and testing it in the 'unseen' test dataset.
    model_results = crf_obj.grid_search_cross_validation(X_train=train_dev_sentences,
                                                         X_test=test_sentences,
                                                         grid_params=grid_params,
                                                         verbose=1,
                                                         k_folds=3)

    # extracting the best model (pos tagger)
    grid_search_best_trained_tagger = model_results['model']

    # running a hands on example for a random testing sentence.
    crf_obj.print_crf_tagger_example(sentence=test_sentences[0])


def main_best_crf_model():
    """
    This function runs the model that performed best in our analysis.

    :return:
    """
    # fetches and creates a dict containing the train, dev and test data.
    data_d = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    # extracting train, dev, and test data sets
    train_dataset = DataFetcher.parse_conllu(data_d['train'])
    dev_dataset = DataFetcher.parse_conllu(data_d['dev'])
    test_dataset = DataFetcher.parse_conllu(data_d['test'])

    # removing any empty sentences (lists) from the data sets.
    train_sentences = DataFetcher.remove_empty_sentences(train_dataset)
    dev_sentences = DataFetcher.remove_empty_sentences(dev_dataset)
    test_sentences = DataFetcher.remove_empty_sentences(test_dataset)

    # concatenating the train and held out (dev) dataset in order to feed it for cross validation.
    train_dev_sentences = train_sentences + dev_sentences

    # Best parameters:
    params = {
        'c1': 0.1,  # coefficient for L1 penalty
        'c2': 0.1,  # coefficient for L2 penalty
        'max_iterations': 200,  # stop earlier
        'feature.possible_transitions': True  # include transitions that are possible, but not observed
    }

    crf_obj = CRFTagger()

    crf_obj.fit(X=train_dev_sentences, params=params, filepath='best_crf_model_en.crfsuite')

    out = crf_obj.evaluate(X=test_sentences, verbose=2)

    return out


def main_benchmark_plot():
    """
    This function creates the learning curves for the best model of our analysis.
    :return:
    """
    # fetches and creates a dict containing the train, dev and test data.
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    # extracting train, dev, and test datasets
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    # removing any empty sentences from the datasets.
    train_sents = DataFetcher.remove_empty_sentences(train_data)
    dev_sents = DataFetcher.remove_empty_sentences(dev_data)
    test_sents = DataFetcher.remove_empty_sentences(test_data)

    # concatenating the train and held out (dev) dataset in order to feed it for cross validation.
    train_dev_sents = train_sents + dev_sents

    # Best parameters:
    params = {'c1': 0.1, 'c2': 0.1, 'max_iterations': 200, 'feature.possible_transitions': True}

    crf_obj = CRFTagger()
    # creating the plot.
    crf_obj.create_benchmark_plot(train=train_dev_sents, test=test_sents, n_splits=20, params=params)


if __name__ == "__main__":
    # running grid search to get the best parameters for our CRF POS Tagger
    main_grid_search()

    # After obtained the best parameters we train the best model
    # and evaluate it with the 'unseen' test dataset
    main_best_crf_model()

    # Creating learning curves for the best model.
    main_benchmark_plot()
