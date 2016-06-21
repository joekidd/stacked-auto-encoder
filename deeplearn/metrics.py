import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

def confusion_matrix(true_y, pred_y, labels):
    c_matrix = metrics.confusion_matrix(true_y, pred_y)

    confusion_table = []
    first_row = ["C.Matrix"] + labels + ["ACTUAL"] + ["RECALL"]
    confusion_table.append(first_row)

    recall = metrics.recall_score(true_y, pred_y, average=None)
    for r, row in enumerate(c_matrix):
        new_row = [labels[r]]
        new_row.extend(row)
        new_row.append(sum(row))
        new_row.append(recall[r])
        confusion_table.append(new_row)

    new_row = ["PREDICTED"]
    for l in labels:
        new_row.append(len([t for t in pred_y if t == l]))
    new_row.append(len(true_y))
    new_row.append(metrics.recall_score(true_y, pred_y))
    confusion_table.append(new_row)

    new_row = ["PRECISION"]
    new_row.extend(metrics.precision_score(true_y, pred_y, average=None))
    new_row.append(metrics.precision_score(true_y, pred_y))
    new_row.append(metrics.f1_score(true_y, pred_y))
    confusion_table.append(new_row)

    confusion_table = pd.DataFrame(confusion_table)
    return confusion_table

