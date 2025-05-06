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
    labels = sorted(set(y_true))
    
    for cls in labels:
        cls_str = str(cls)
        cls_indices = [i for i, y in enumerate(y_true) if y == cls]
        y_true_cls = [y_true[i] for i in cls_indices]
        y_pred_cls = [y_pred[i] for i in cls_indices]
        
        acc = accuracy_score(y_true_cls, y_pred_cls)
        f1 = report[cls_str]['f1-score']

        metrics[f'class_{cls}_accuracy'] = acc
        metrics[f'class_{cls}_f1'] = f1
        metrics[f'class_{cls}_precision'] = report[cls_str]['precision']
        metrics[f'class_{cls}_recall'] = report[cls_str]['recall']
    
    return metrics
