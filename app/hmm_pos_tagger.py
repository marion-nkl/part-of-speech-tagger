from pprint import pprint
import numpy as np
from nltk.corpus import treebank
from operator import add

from app.data_fetcher import DataFetcher
from app.evaluation import tagger_classification_report
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


class HMMTagger:
    def __init__(self):
        # p(state | state-1) = count(state-1, state) / count(state-1)
        self.transition_probabilities = dict()

        # p(word | state) = count(state, word) / count(state)
        self.emission_probabilities = dict()
        self.transition_probability_matrix = dict()

        self._state_frequencies = dict()
        self._word_frequencies = dict()
        self.states = set()
        self._train_data = None
        self.viterbi = dict()

        self.final_hmm = {0: {'<start>': {'argmax': None, 'viterbi': 2},
                              'DT': {'argmax': None, 'viterbi': 0.5},
                              'NNP': {'argmax': None, 'viterbi': 0},
                              'VBZ': {'argmax': None, 'viterbi': 1}},
                          1: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                              'DT': {'argmax': 'NNP', 'viterbi': 0.0},
                              'NNP': {'argmax': 'DT', 'viterbi': 0.0},
                              'VBZ': {'argmax': 'DT', 'viterbi': 1.0}},
                          2: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                              'DT': {'argmax': 'NNP', 'viterbi': 2.0},
                              'NNP': {'argmax': 'NNP', 'viterbi': 0.0},
                              'VBZ': {'argmax': 'NNP', 'viterbi': 1.0}},
                          3: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                              'DT': {'argmax': 'NNP', 'viterbi': 2.0},
                              'NNP': {'argmax': 'NNP', 'viterbi': 0.0},
                              'VBZ': {'argmax': 'NNP', 'viterbi': 0.0}}}

    @staticmethod
    def _pad_sentence(sentence):
        """
        Pad a sentence in order to add starting and end tokens in each sentence
        :param sentence: list of tuples with the word of each sentence along with its POS tag
        :return: list of tuples, a padded sentence with start and end tokens
        """

        sentence.insert(0, ('<start>', '<start>'))
        sentence.insert(len(sentence), ('<end>', '<end>'))

        return sentence

    @staticmethod
    def extract_pos_tags_from_sentence_token_tuples(sent):
        """
        This function extracts the pos tags (labels in general) from a given iterable of (token, label) tuples.

        :param sent: list. An iterable of (word, pos-tag) tuples.
        :return: list. An iterable of pos tags.
        """
        return [postag for token, postag in sent]

    def fit(self, data):
        """
        Creates two probability dictionaries storing the POS-to-POS and the POS-to-WORD probabilities
        :param data: list of lists of tuples, with sentences and their words with their POS tags
        """
        self._train_data = data

        for sentence in self._train_data:
            new_sentence = self._pad_sentence(sentence)
            for i in range(len(new_sentence)):
                self.states.add(new_sentence[i][1])

                try:
                    self._state_frequencies[new_sentence[i][1]] += 1
                except KeyError:
                    self._state_frequencies[new_sentence[i][1]] = 1

                try:
                    self._word_frequencies[new_sentence[i][0]] += 1
                except KeyError:
                    self._word_frequencies[new_sentence[i][0]] = 1

                try:
                    self.transition_probabilities[new_sentence[i][1], new_sentence[i + 1][1]] += 1
                except KeyError:
                    self.transition_probabilities[new_sentence[i][1], new_sentence[i + 1][1]] = 1
                except IndexError:
                    pass
                try:
                    self.emission_probabilities[new_sentence[i][1], new_sentence[i][0]] += 1
                except KeyError:
                    self.emission_probabilities[new_sentence[i][1], new_sentence[i][0]] = 1

        for state_pair in self.transition_probabilities:
            self.transition_probabilities[state_pair] = self.transition_probabilities[state_pair] / \
                                                        self._state_frequencies[state_pair[0]]

        for state_word_pair in self.emission_probabilities:
            self.emission_probabilities[state_word_pair] = self.emission_probabilities[state_word_pair] / \
                                                           self._state_frequencies[state_word_pair[0]]

        self.states.remove('<start>')
        self.states.remove('<end>')

    @staticmethod
    def _find_max(viterbi_previous, transition_prob):
        """
        Find the max value kai the max arg for a matrix (output of an element-wise multiplication between previous
        viterbi probabilities: v, and transitions probabilities: a
        :param viterbi_previous: list with viterbi log probabilities of the previous state
        :param transition_prob: list with transition probabilities for each previous state to the current
        :return: the max value and max arg
        """
        transition_prob_log = list(np.log(transition_prob))
        max_value = np.max(np.add(viterbi_previous, transition_prob_log))
        max_state = np.argmax(np.add(viterbi_previous, transition_prob_log))
        return max_value, max_state

    def _viterbi(self, sequence):
        """
        Run forward pass of the Viterbi algorithm
        :param sequence: list of tuples with words and their tags
        """
        word = sequence[0][0]
        self.viterbi[0] = dict()

        # initialization
        for state in self.states:
            self.viterbi[0][state] = dict()
            a = self.transition_probabilities.get(('<start>', state), 1)
            b = self.emission_probabilities.get((state, word), 1)

            self.viterbi[0][state]['viterbi'] = np.log(a) + np.log(b)
            self.viterbi[0][state]['argmax'] = None

        # fill in for the rest of the steps /words
        for i in range(1, len(sequence) + 1):
            if i != (len(sequence)):
                self.viterbi[i] = dict()
                for state in self.states:
                    self.viterbi[i][state] = dict()
                    b = self.emission_probabilities.get((state, sequence[i][0]), 1)

                    # For each previous state find the max viterbi
                    previous_viterbi_list = list()
                    previous_states = list()
                    current_transitions = list()

                    for state_prev in self.viterbi[i - 1]:
                        if state_prev != '<start>':
                            previous_states.append(state_prev)
                            previous_viterbi_list.append(self.viterbi[i - 1][state_prev]['viterbi'])
                            current_transitions.append(self.transition_probabilities.get((state_prev, state), 1))

                    v_prev, state_prev = self._find_max(previous_viterbi_list, current_transitions)

                    # fill table
                    self.viterbi[i][state]['viterbi'] = np.log(b) + v_prev
                    self.viterbi[i][state]['argmax'] = previous_states[state_prev]

            else:
                # final step
                self.viterbi[i] = dict()

                previous_viterbi_list = list()
                previous_states = list()
                current_transitions = list()
                for state in self.viterbi[i - 1]:
                    self.viterbi[i]['<end>'] = dict()

                    previous_states.append(state)
                    previous_viterbi_list.append(self.viterbi[i - 1][state]['viterbi'])
                    current_transitions.append(self.transition_probabilities.get((state, '<end>'), 1))

                v_prev, state_prev = self._find_max(previous_viterbi_list, current_transitions)

                self.viterbi[i]['<end>']['viterbi'] = v_prev
                self.viterbi[i]['<end>']['argmax'] = previous_states[state_prev]

    def _get_final_path(self):
        """
        Returns the backtrace path by following backpointers to states back in time
        :return: list, with the predicted path for the viterbi matrix
        """
        # best_end, state_prev = self._find_max(previous_viterbi_list, current_transitions)

        final_path = list()
        for s in range(len(self.viterbi) - 1, -1, -1):
            state_list = list()
            viterbi_p = list()
            transition_a = list()
            for state in self.viterbi[s]:
                state_list.append(state)
                viterbi_p.append(self.viterbi[s][state]['viterbi'])
                transition_a.append(self.transition_probabilities.get((s, '<end>'), 1))

            position = np.argmax(viterbi_p + np.log(transition_a))
            final_path.append(state_list[position])

        # swap list
        true_path = list()
        for i in range(len(final_path) - 1, -1, -1):
            true_path.append(final_path[i])

        return true_path

    def tag(self, sentence):
        """
        Implements Viterbi algorithm to find the hidden state sequence of the observed sequence
        :param sentence: list of tuples with words and their tags
        :return: list, with the predicted path for the viterbi matrix
        """
        self._viterbi(sentence)

        path = self._get_final_path()
        # return all the path but not the <end> final state
        return path[:-1]

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
            self.fit(train_x_part)

            # checking the results always on the same test set
            test_labels = [self.extract_pos_tags_from_sentence_token_tuples(x) for x in test]

            print(test_labels)

            predicted_labels = list()
            for sent in test:
                predicted_labels.append(self.tag(sent))

            print(predicted_labels)

            result_on_test = tagger_classification_report(test_labels, predicted_labels)
            results['on_test'].append(result_on_test['accuracy'])

            # calculates the metrics for the given training part
            train_labels = [self.extract_pos_tags_from_sentence_token_tuples(x) for x in train]

            predicted_labels = list()
            for sent in train:
                predicted_labels.append(self.tag(sent))

            result_on_train_part = tagger_classification_report(train_labels, predicted_labels)
            results['on_train'].append(result_on_train_part['accuracy'])

            line_up, = ax.plot(results['train_size'], results['on_train'], 'o-', label='Accuracy on Train')
            line_down, = ax.plot(results['train_size'], results['on_test'], 'o-', label='Accuracy on Test')

            plt.legend([line_up, line_down], ['Accuracy on Train', 'Accuracy on Test'], prop=font_p)

        if plot_outfile:
            fig.savefig(plot_outfile)

        plt.show()

        return results


if __name__ == '__main__':
    # # create a dict with pos-to-pos probabilities and pos-to-word probabilities on the training set
    sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP'), ('.', 'PUNCT')],
                 [('That', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    data_dict = DataFetcher.read_data()
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    cleaned_train_data = DataFetcher.remove_empty_sentences(train_data + dev_data)

    data = treebank.tagged_sents()[:3000]

    tagger = HMMTagger()
    # tagger.fit(cleaned_train_data)

    # pprint(tagger.emission_probabilities)
    # print()
    # pprint(tagger.transition_probabilities)

    test_sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP'), ('.', 'PUNCT')],
                      [('That', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    test_data = DataFetcher.parse_conllu(data_dict['test'])
    cleaned_test_data = DataFetcher.remove_empty_sentences(test_data)

    # Tag sentences
    results = list()
    # for s in test_sentences:
        # results.append(tagger.tag(s))

    print()
    pprint(results)

    # pprint(tagger_classification_report(results, results))

    tagger.create_benchmark_plot(cleaned_train_data, cleaned_test_data)




    # ------------------------------------------------------------------
    # # train data
    # data_dict = DataFetcher.read_data()
    # train_data = DataFetcher.parse_conllu(data_dict['train'])
    # dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    # cleaned_train_data = DataFetcher.remove_empty_sentences(train_data + dev_data)
    # # fit HMM model
    # tagger = HMMTagger()
    # tagger.fit(cleaned_train_data)
    #
    # test_data = DataFetcher.parse_conllu(data_dict['test'])
    # cleaned_test_data = DataFetcher.remove_empty_sentences(test_data)
    #
    # # Tag sentences
    # results = list()
    # for s in cleaned_test_data:
    #     results.append(tagger.tag(s))
    #
    # # evaluation report
    # pprint(tagger_classification_report(results, results))
