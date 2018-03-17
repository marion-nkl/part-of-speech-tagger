import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def create_report(y_true, y_pred, classes=None):
    """
    This function calculates several metrics about a classifier and creates a mini report.
    :param y_true: iterable. An iterable of string or ints.
    :param y_pred: iterable. An iterable of string or ints.
    :param classes: iterable. An iterable of string or ints.
    :return: dataframe. A pandas dataframe with the confusion matrix.
    """
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred),
                             index=classes,
                             columns=['pred_{}'.format(c) for c in classes]
                             )

    print("-" * 80, end='\n')
    print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("-" * 80)

    print("Confusion Matrix:", end='\n\n')
    print(confusion)

    print("-" * 80, end='\n')
    print("Classification Report:", end='\n\n')
    print(classification_report(y_true, y_pred, digits=3), end='\n')

    return confusion
