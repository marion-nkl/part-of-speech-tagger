from pprint import pprint
import numpy as np

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

        :param data:
        :return:
        """
        self._train_data = data

        for sentence in self._train_data:
            new_sentence = self._pad_sentence(sentence)
            for i in range(len(new_sentence) - 1):

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

    @staticmethod
    def _viterbi_path(current_state_prob, viterbi_previous, transition_prob):
        """

        :param current_state_prob:
        :param viterbi_previous:
        :param transition_prob:
        :return:
        """
        viterbi_current = current_state_prob * np.max(np.multiply(viterbi_previous, transition_prob))

        return viterbi_current

    def function_rec(self, sentence):

        def return_max(x):
            # print('Max returned {}'.format(x))
            pass

        def fill_in(x, viterbi_previous, transition_prob):
            if x > 10:
                pass
            else:
                v_j = self._viterbi_path(5, viterbi_previous, transition_prob)

                fill_in(x + 1, viterbi_previous, transition_prob)
                return_max(x)

        matrix = dict()

        for word in sentence:
            v_p = np.array([1, 2, 3])
            t_p = np.array([1, 2, 3])
            matrix[word[0]] = dict()
            fill_in(0, v_p, t_p)

        print(matrix)

    def viterbi(self, sentence):
        """

        :param sentence:
        :return:
        """
        viterbi = dict()
        for state in self.states:
            viterbi[state] = self.emission_probabilities[(state, sentence[0][0])] * self.transition_probabilities[
                ('<start>', state)]
            print('Emission from {} to {}: {}'.format(state, sentence[0][0],
                                                      self.emission_probabilities[(state, sentence[0][0])]))

    def _run_decoder(self, sentence):
        """
        Implements Viterbi algorithm to find the hidden state sequence of the observed sequence
        :param sentence:
        :return:
        """
        states_list = list(self.states)
        for word in self._pad_sentence(sentence):
            print()
            print('WORD: {}'.format(word[0]))
            for state in self.states:
                print(state)
                print('State observation likelihood of tag {} to word {}: {}'.format(state, word[0],
                                                                                     self.emission_probabilities.get(
                                                                                         (state, word[0]), 0)))
                for previous_state in self.states:
                    print('Transition probability of tag {} to tag {}: {}'.format(previous_state, state,
                                                                                  self.transition_probabilities.get(
                                                                                      (previous_state, state), 0)))

    def tag(self, sentence):
        self._run_decoder(sentence)


if __name__ == '__main__':
    # # create a dict with pos-to-pos probabilities and pos-to-word probabilities on the training set
    sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    data_dict = DataFetcher.read_data()
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    cleaned_train_data = DataFetcher.remove_empty_sentences(train_data)

    tagger = HMMTagger()
    tagger.fit(sentences)
    #
    # pprint(tagger.emission_probabilities)
    # print()
    # pprint(tagger.transition_probabilities)
    #
    test_sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                      [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]
    #
    # # run viterbi algorithm ont he test set to tag it
    # for sent in test_sentences:
    #     tagger.tag(sent)

    # tagger.viterbi(test_sentences[0])

    emission = tagger.emission_probabilities
    transition = tagger.transition_probabilities
    all_states = tagger.states


    def find_max(viterbi_previous, transition_prob):
        """

        :param viterbi_previous:
        :param transition_prob:
        :return:
        """
        v = np.array(viterbi_previous)
        t = np.array(transition_prob)
        max_value = np.max(np.multiply(v, t))
        max_state = np.argmax(np.multiply(v, t))
        return max_value, max_state


    viterbi = dict()
    word = test_sentences[0][0][0]
    viterbi[0] = dict()

    pprint(emission)
    print()
    pprint(transition)
    print()

    # initialization
    for state in all_states:
        viterbi[0][state] = dict()
        viterbi[0][state]['argmax'] = None
        # max_v, max_s = find_max()
        a = transition.get(('<start>', state), 0)
        b = emission.get((state, word), 0)
        viterbi[0][state]['viterbi'] = a * b

    print()

    # fill in
    sentence = test_sentences[0]
    for i in range(1, len(test_sentences[0])):
        viterbi[i] = dict()
        for state in all_states:
            viterbi[i][state] = dict()
            b = emission.get((state, word), 0)

            # For each previous state find the max viterbi
            previous_viterbi_list = list()
            previous_states = list()
            current_transitions = list()
            for state_prev in viterbi[i - 1]:
                previous_states.append(state_prev)
                previous_viterbi_list.append(viterbi[i - 1][state_prev]['viterbi'])
                current_transitions.append(transition.get((state_prev, state), 0))

            # find max of previous viterbi path and transition probs
            v_prev, state_prev = find_max(previous_viterbi_list, current_transitions)

            # fill table
            viterbi[i][state]['viterbi'] = b * v_prev
            viterbi[i][state]['argmax'] = previous_states[state_prev]

    print()
    pprint(viterbi)

    final_hmm = {0: {'<start>': {'argmax': None, 'viterbi': 0},
                     'DT': {'argmax': None, 'viterbi': 0.5},
                     'NNP': {'argmax': None, 'viterbi': 0},
                     'VBZ': {'argmax': None, 'viterbi': 0}},
                 1: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                     'DT': {'argmax': 'NNP', 'viterbi': 0.0},
                     'NNP': {'argmax': 'DT', 'viterbi': 0.0},
                     'VBZ': {'argmax': 'DT', 'viterbi': 0.0}},
                 2: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                     'DT': {'argmax': 'NNP', 'viterbi': 0.0},
                     'NNP': {'argmax': 'NNP', 'viterbi': 0.0},
                     'VBZ': {'argmax': 'NNP', 'viterbi': 0.0}},
                 3: {'<start>': {'argmax': 'NNP', 'viterbi': 0.0},
                     'DT': {'argmax': 'NNP', 'viterbi': 0.0},
                     'NNP': {'argmax': 'NNP', 'viterbi': 0.0},
                     'VBZ': {'argmax': 'NNP', 'viterbi': 0.0}}}

    for s in range(len(final_hmm), -1, -1):
        print(s)


