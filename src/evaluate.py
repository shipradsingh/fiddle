import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

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
    acc = accuracy_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_bin, average='binary')

    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
