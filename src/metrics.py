import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    sensitivity = []
    specificity = []

    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity.append(tp / (tp + fn) if tp + fn else 0.0)
        specificity.append(tn / (tn + fp) if tn + fp else 0.0)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "sensitivity_per_class": dict(zip(class_names, sensitivity)),
        "specificity_per_class": dict(zip(class_names, specificity)),
    }
    return result
