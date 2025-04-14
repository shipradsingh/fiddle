import torch
import numpy as np
import sklearn.metrics as metrics

"""
Evaluate the model on the validation or test set.
Calculates all core classification metrics and prints them.
Assumes model outputs probabilities, so we threshold at 0.5 to get binary predictions.
"""
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, label in dataloader:
            pred = model(x1, x2).squeeze().numpy()
            all_preds.extend(pred)
            all_labels.extend(label.numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    acc = metrics.accuracy_score(all_labels, preds_bin)
    auc = metrics.roc_auc_score(all_labels, all_preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(all_labels, preds_bin, average='binary')

    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
