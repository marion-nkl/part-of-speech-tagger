import itertools as it

import numpy as np
from numpy import random


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
