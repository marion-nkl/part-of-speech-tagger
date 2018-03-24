from pprint import pprint

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
                    self.emission_probabilities[new_sentence[i][1], new_sentence[i + 1][0]] += 1
                except KeyError:
                    self.emission_probabilities[new_sentence[i][1], new_sentence[i + 1][0]] = 1

        for state_pair in self.transition_probabilities:
            self.transition_probabilities[state_pair] = self.transition_probabilities[state_pair] / self._state_frequencies[state_pair[0]]

        for state_word_pair in self.emission_probabilities:
            self.emission_probabilities[state_word_pair] = self.emission_probabilities[state_word_pair] / self._state_frequencies[state_word_pair[0]]

    def _run_viterbi(self, sentence):
        pass

    def tag(self, sentence):
        self._run_viterbi(sentence)


if __name__ == '__main__':

    # create a dict with pos-to-pos probabilities and pos-to-word probabilities on the training set
    sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                 [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    data_dict = DataFetcher.read_data()
    train_data = DataFetcher.parse_conllu(data_dict['train'])
    cleaned_train_data = DataFetcher.remove_empty_sentences(train_data)

    tagger = HMMTagger()
    tagger.fit(cleaned_train_data)

    pprint(tagger.state_to_state)

    test_sentences = [[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')],
                      [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('sentence', 'NNP')]]

    # run viterbi algorithm ont he test set to tag it
    for sent in test_sentences:
        tagger.tag(sent)
