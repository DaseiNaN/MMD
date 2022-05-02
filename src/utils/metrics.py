import numpy as np
from sklearn.metrics import confusion_matrix


def standard_confusion_matrix(y_pred, y_true):
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
    return np.array([[tp, fp], [fn, tn]]).astype(np.float32)


def measure_performance(y_pred, y_true):
    conf_matrix = standard_confusion_matrix(y_true, y_pred.numpy())
    [[tp, fp], [fn, tn]] = conf_matrix
    accuracy = (tp + tn) / np.sum(conf_matrix)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return conf_matrix, [accuracy, precision, recall, f1_score]


if __name__ == "__main__":
    import torch

    y_pred = torch.randn((5, 2))
    print(y_pred)
    print(y_pred.max(dim=1, keepdim=True)[1])
