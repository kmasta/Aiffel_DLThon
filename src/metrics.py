from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def category_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {}
    for cls, vals in report.items():
        if cls.isdigit():
            metrics[f'class_{cls}_accuracy'] = vals['accuracy']
            metrics[f'class_{cls}_f1'] = vals['f1-score']
    return metrics
