## Part of speech tagger

This project is a part of the third assignment for the Text engineering course of the program: MSc in Data Science (AUEB). It develops a sequence POS tagger for the english language of the Universal Dependencies treebanks using HMM and CRF models.


### Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Install the requirements.txt
2. run the hmm_pos_tagger.py (for modeling and evaluation of hmm tagger) or  crf_pos_tagger.py (for modeling, cross validation and evaluation of crf tagger).

### Installing
In order to run the code in your local environment, please make sure your have python 3. and above and to have installed the needed python libraries. To install the libraries please run on your console:

```
pip install -r requirements.txt
```

### Train a POS tagger
In order to train a POS tagger with the custom HMM algorithm you will need to run the following command:

```
python app/hmm_pos_tagger.py
```

In order to train a POS tagger with the custom CRF algorithm you will need to run the following command:

```
python app/crf_pos_tagger.py
```

In order to train a POS tagger with a baseline algorithm you will need to run the following command:

```
python app/baseline.py
```

In order to compare POS taggers trained with `nltk` HMM and CRF implementations, you will need to run the following command:

```
python app/pos_taggers.py
```

### Structure
The project consists of the following main classes:

- [DataFetcher](https://github.com/marion-nkl/part-of-speech-tagger/blob/master/app/data_fetcher.py)
- [HMMTagger](https://github.com/marion-nkl/part-of-speech-tagger/blob/master/app/hmm_pos_tagger.py)
- [CRFTagger](https://github.com/marion-nkl/part-of-speech-tagger/blob/master/app/crf_pos_tagger.py)
- [Baseline](https://github.com/marion-nkl/part-of-speech-tagger/blob/master/app/baseline.py)

and [evaluation.py](https://github.com/marion-nkl/part-of-speech-tagger/blob/master/app/evaluation.py)

#### Data Fetcher
This class is responsible for all the handling and fetching of the dataset(s). It loads the data, parse `.conllu` formatting and creates a label dataset that will be fed in the models. 

#### HMMTagger
This class implements a HMMTagger model and test its custom Viterbi decoder implementation. It runs cross validations in the train and test datasets and yields the benchmark results.

#### CRFTagger
This class implements a CRFTagger model. It creates features for the tokens in the dataset, runs cross validations in the train and test datasets and yields the benchmark results.

#### Baseline
This class implements a baseline model which taggs each word with the most frequent tag it had in the training set. 

#### evaluation.py
This file contains util functions that are responsible for the evaluation of a given model. It calculates the accuracy and F1 scores for the all the classes as well as each class separately.




