from pprint import pprint
import numpy as np
from nltk.corpus import treebank
from operator import add

from app.data_fetcher import DataFetcher


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
        print('First step')
        for state in self.states:
            self.viterbi[0][state] = dict()
            a = self.transition_probabilities.get(('<start>', state), 1)
            b = self.emission_probabilities.get((state, word), 1)

            self.viterbi[0][state]['viterbi'] = np.log(a) + np.log(b)
            self.viterbi[0][state]['argmax'] = None
        print('-'*40)

        # fill in for the rest of the steps /words
        for i in range(1, len(sequence)+1):
            print('Step: {}'.format(i))
            if i != (len(sequence)):
                self.viterbi[i] = dict()
                for state in self.states:
                    print('State: {}'.format(state))
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
                    print('Viterbi: {}',format(self.viterbi[i][state]['viterbi']))

            else:
                # final step
                self.viterbi[i] = dict()

                previous_viterbi_list = list()
                previous_states = list()
                current_transitions = list()
                for state in self.viterbi[i - 1]:
                    self.viterbi[i][state] = dict()

                    previous_states.append(state)
                    previous_viterbi_list.append(self.viterbi[i - 1][state]['viterbi'])
                    current_transitions.append(self.transition_probabilities.get((state, '<end>'), 1))

                    v_prev, state_prev = self._find_max(previous_viterbi_list, current_transitions)

                    self.viterbi[i][state]['viterbi'] = v_prev
                    self.viterbi[i][state]['argmax'] = previous_states[state_prev]

    def _get_final_path(self):
        """
        Returns the backtrace path by following backpointers to states back in time
        :return: list, with the predicted path for the viterbi matrix
        """
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
        return path


if __name__ == '__main__':
    # # create a dict with pos-to-pos probabilities and pos-to-word probabilities on the training set
    sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP'), ('.', 'PUNCT')],
                 [('That', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    data_dict = DataFetcher.read_data()
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    cleaned_train_data = DataFetcher.remove_empty_sentences(train_data)

    data = treebank.tagged_sents()[:3000]

    tagger = HMMTagger()
    tagger.fit(cleaned_train_data)

    # pprint(tagger.emission_probabilities)
    # print()
    # pprint(tagger.transition_probabilities)

    test_sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP'), ('.', 'PUNCT')],
                      [('That', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    print()
    print(tagger.tag(test_sentences[0]))

    print()
    pprint(tagger.viterbi)
