import numpy as np
from sklearn.metrics import confusion_matrix


def standard_confusion_matrix(y_true, y_pred):
    """Standard confusion matrix

    ====================================
    |                |      Truth      |
    |----------------+-----------------|
    |                |   True   False  |
    | Pred  Positive |    TP     FP    |
    |       Negative |    FN     TN    |
    ====================================
    """

    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def measure_performance(y_true, y_pred):
    y_pred = y_pred.max(dim=1, keepdim=True)[1]
    conf_matrix = standard_confusion_matrix(y_true, y_pred.numpy())
    return y_pred, conf_matrix
