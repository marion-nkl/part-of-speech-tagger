import os
from collections import Counter

import pandas as pd
import pycrfsuite
from numpy import mean

pd.set_option('display.expand_frame_repr', False)

from app import MODELS_DIR
from app.data_fetcher import DataFetcher
from app.evaluation import crf_tagger_classification_report, print_crf_transitions
from app.utils import feed_crf_params, feed_cross_validation

CRF_MODEL_FILEPATH = os.path.join(MODELS_DIR, 'crf_model_en.crfsuite')
GRID_SEARCH_CRF_MODEL_FILEPATH = os.path.join(MODELS_DIR, 'temp_crf_model_en.crfsuite')


def get_word_features(sent, position):
    """

    :param sent:
    :param position:
    :return:
    """

    # extracting the word
    word = sent[position][0]

    features = [
        'bias',
        'word.lower={}'.format(word.lower()),  # the word in lower
        'last_three_chars={}'.format(word[-3:]),  # up to last three characters
        'last_two_chars={}'.format(word[-2:]),  # up to last two characters
        'first_two_chars={}'.format(word[:2]),  # up to first two characters
        'first_three_chars={}'.format(word[:2]),  # up dto frist three characters
        'word.isupper={}'.format(word.isupper()),  # checks whether the word is written in uppercase.
        'word.istitle={}'.format(word.istitle()),  # checks if the first letter is in uppercase.
        'word.isdigit={}'.format(word.isdigit()),  # checks whether it is a digit.
        'word.endswith.ed={}'.format(word.endswith('ed')),
        'word.endswith.ing={}'.format(word.endswith('ing'))
    ]

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


def get_sentence_to_features(sent):
    """

    :param sent:
    :return:
    """
    return [get_word_features(sent, num) for num in range(len(sent))]


def extract_labels_from_sentence_token_tuples(sent):
    """

    :param sent:
    :return:
    """
    return [postag for token, postag in sent]


def extract_tokens_from_sentence_token_tuples(sent):
    """

    :param sent:
    :return:
    """
    return [token for token, postag in sent]


def train_crf_model(training_sentences,
                    test_sentences,
                    params=None,
                    verbose=0,
                    filepath=CRF_MODEL_FILEPATH,
                    load_model=False):
    """

    :param training_sentences:
    :param test_sentences:
    :param params:
    :param verbose:
    :param filepath:
    :param load_model:
    :return:
    """
    if load_model:

        tagger = pycrfsuite.Tagger()
        tagger.open(filepath)

    else:

        if params is None:
            # Set training parameters. We will use L-BFGS training algorithm (it is default)
            # with Elastic Net (L1 + L2) regularization.
            params = {
                'c1': 1.0,  # coeff for L1 penalty
                'c2': 1e-3,  # coeff for L2 penalty
                'max_iterations': 50,  # stop earlier
                'feature.possible_transitions': True  # include transitions that are possible, but not observed
            }

        # extracting the features and the labels (pos tags) for the train examples
        X_train = [get_sentence_to_features(s) for s in training_sentences]
        y_train = [extract_labels_from_sentence_token_tuples(s) for s in training_sentences]

        # Training the model:
        # In order to train the model, we create a pycrfsuite.Trainer, load the training data and calling the 'train' method

        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        # setting the parameters for the models.
        trainer.set_params(params)
        trainer.train(filepath)

        # Make predictions
        # In order to use the trained model, we create a pycrfsuite.Tagger, open the model and use "tag" method:
        tagger = pycrfsuite.Tagger()
        tagger.open(filepath)

    model_accuracy = None
    if test_sentences:

        X_test = [get_sentence_to_features(s) for s in test_sentences]
        y_test = [extract_labels_from_sentence_token_tuples(s) for s in test_sentences]
        # Predicting pos tag labels for all sentences in our testing set
        y_pred = [tagger.tag(xseq) for xseq in X_test]

        model_res = crf_tagger_classification_report(y_test, y_pred)

        model_accuracy = model_res['accuracy']
        model_clf_report = model_res['clf_report']

        if verbose > 0:
            # Checking what classifier has learned
            info = tagger.info()

            print('Model Accuracy: {}'.format(model_accuracy), end='\n\n')

            print(model_clf_report)

            print("Top likely Pos Tags transitions:")
            print_crf_transitions(Counter(info.transitions).most_common(15))

            print("\nTop unlikely Pos Tags transitions:")
            print_crf_transitions(Counter(info.transitions).most_common()[-15:])

    return {'model': tagger, 'accuracy': model_accuracy}


def print_crf_tagger_example(trained_tagger, sentence):
    """

    :param trained_tagger:
    :param sentence:
    :return:
    """

    print('\n\nSentence: "{}"'.format(' '.join(extract_tokens_from_sentence_token_tuples(sentence)), end='\n\n'))

    print("Predicted:", ' '.join(trained_tagger.tag(get_sentence_to_features(sentence))))
    print("Correct:  ", ' '.join(extract_labels_from_sentence_token_tuples(sentence)))


def get_grid_search_crf_model(training_sentences,
                              test_sentences,
                              grid_params=None,
                              verbose=0,
                              k_folds=3,
                              filepath=GRID_SEARCH_CRF_MODEL_FILEPATH):
    """

    :param training_sentences:
    :param test_sentences:
    :param grid_params:
    :param verbose:
    :param k_folds:
    :param filepath:
    :return:
    """

    param_combinations = feed_crf_params(grid_search_params=grid_params)

    grid_search_results = list()

    for param_combination in param_combinations:
        d = dict()
        d['parameters'] = param_combination

        scores = list()

        if verbose > 0: print('Model Parameters: {}'.format(param_combination))

        for num, data in enumerate(feed_cross_validation(sentences=training_sentences, k_folds=k_folds)):
            train = data['train']
            held_out = data['held_out']

            verb = 1 if verbose > 1 else 0
            model_meta = train_crf_model(training_sentences=train,
                                         test_sentences=held_out,
                                         params=param_combination,
                                         verbose=verb,
                                         filepath=filepath)

            d["fold{}_score".format(num + 1)] = model_meta['accuracy']
            scores.append(model_meta['accuracy'])

        d['mean_test_score'] = mean(scores)

        grid_search_results.append(d)

    best_model_metadata = sorted(grid_search_results, key=lambda x: x['mean_test_score'], reverse=True)

    if verbose == 1:
        print(pd.DataFrame(grid_search_results))

    best_parameters = best_model_metadata[0]['parameters']

    best_model_meta = train_crf_model(training_sentences=training_sentences,
                                      test_sentences=test_sentences,
                                      params=best_parameters,
                                      verbose=verbose,
                                      filepath=filepath)

    return best_model_meta


if __name__ == "__main__":
    # fetches and creates a dict containing the train, dev and test data.
    data_dict = DataFetcher.read_data(files_list=['train', 'dev', 'test'])

    train_data = DataFetcher.parse_conllu(data_dict['train'])
    dev_data = DataFetcher.parse_conllu(data_dict['dev'])
    test_data = DataFetcher.parse_conllu(data_dict['test'])

    train_sents = DataFetcher.remove_empty_sentences(train_data)
    dev_sents = DataFetcher.remove_empty_sentences(dev_data)
    test_sents = DataFetcher.remove_empty_sentences(test_data)

    train_dev_sents = train_sents + dev_sents

    grid_params = {
        'c1': [1.0, 0.1, 0.01],  # coeff for L1 penalty
        'c2': [1.0, 0.1, 0.01],  # coeff for L2 penalty
        'max_iterations': [50, 100, 200],  # stop earlier
        'feature.possible_transitions': [True, False]  # include transitions that are possible, but not observed
    }

    model_results = get_grid_search_crf_model(training_sentences=train_dev_sents,
                                              test_sentences=test_sents,
                                              grid_params=grid_params,
                                              k_folds=3,
                                              verbose=1)

    grid_search_best_trained_tagger = model_results['model']

    example_sent = test_sents[0]
    print_crf_tagger_example(trained_tagger=grid_search_best_trained_tagger,
                             sentence=dev_sents[0])
