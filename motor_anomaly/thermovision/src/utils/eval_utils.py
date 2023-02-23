from typing import List

from sklearn import metrics
from tabulate import tabulate


def calculate_metrics_per_class(y_true: List[str], y_pred: List[str], class_mapping: List[str]):
    """Calculates metrics per class (precision, recall and F1 score) and prints them."""
    total_predictions = [0] * len(class_mapping)
    total_true = [0] * len(class_mapping)

    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_mapping)

    for i in range(len(class_mapping)):
        for j in range(len(class_mapping)):
            total_predictions[i] += cm[j][i]
            total_true[i] += cm[i][j]

    # data for confusion matrix
    cm_data = [["", *class_mapping, "total"]]
    for row_idx, class_name in enumerate(class_mapping):
        row = [class_name, *[cm[row_idx][col_idx] for col_idx in range(len(class_mapping))], total_true[row_idx]]
        cm_data.append(row)

    cm_data.append(["Total predicted:", *[total_predictions[idx] for idx in range(len(class_mapping))], sum(total_true)])

    precisions, recalls, f1_scores = [], [], []
    for idx, class_name in enumerate(class_mapping):
        precision = cm[idx][idx] / total_true[idx]
        recall = cm[idx][idx] / total_predictions[idx]

        precisions.append(round(precision, 2))
        recalls.append(round(recall, 2))
        f1_scores.append(round((2 * precision * recall) / (precision + recall), 2))

    metrics_data = [
        ["", *class_mapping],
        ["F1 Score", *f1_scores],
        ["Recall", *recalls],
        ["Precision", *precisions]
    ]

    # printing confusion matrix, f1 score, recall and precision
    print(tabulate(cm_data, tablefmt="simple_grid"))
    print(tabulate(metrics_data, tablefmt="simple_grid"))
