from itertools import chain

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer


def create_report(y_true, y_pred, classes=None):
    """
    This function calculates several metrics about a classifier and creates a mini report.
    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    if classes is None:
        confusion = pd.DataFrame(confusion_matrix(y_true, y_pred))

    else:
        confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                                 index=classes,
                                 columns=['pred_{}'.format(c) for c in classes])

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion


def crf_tagger_classification_report(y_true, y_pred):
    """
    Classification report for a list of pos-tags-encoded sequences. It computes token-level metrics

    :param y_true:
    :param y_pred:
    :return:
    """
    lb = LabelBinarizer()

    # flattens the results for the list of lists of tuples
    y_true_flat = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_flat = lb.transform(list(chain.from_iterable(y_pred)))

    pos_tags_set = sorted(set(lb.classes_))
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')

    clf_report = classification_report(
        y_true_flat,
        y_pred_flat,
        digits=3,
        labels=[class_indices[cls] for cls in pos_tags_set],
        target_names=pos_tags_set)

    return {'accuracy': accuracy,
            'clf_report': clf_report,
            'y_true_flat': y_true_flat,
            'y_pred_flat': y_pred_flat,
            'y_true_size': len(y_true),
            'y_pred_size': len(y_pred),
            'f1': f1}


def print_crf_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
