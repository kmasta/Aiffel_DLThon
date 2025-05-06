import numpy as np
from joblib import load, dump


def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def proba(model, texts):
    """
    Return prediction probabilities for given texts.
    If model has predict_proba, use it; otherwise apply softmax to model.predict.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    return softmax(model.predict(texts), axis=-1)


def get_preds(models, weights, strategy, meta_path, texts):
    """
    Ensemble inference helper.

    Args:
        models (List): list of model instances
        weights (List[float]): ensemble weights for soft voting
        strategy (str): 'weighted_soft' or 'stacking'
        meta_path (str): path to saved meta-model (for stacking)
        texts (List[str]): input texts to predict

    Returns:
        List: predicted labels
    """
    if strategy.startswith("weighted"):
        # soft voting
        probs = [w * proba(m, texts) for m, w in zip(models, weights)]
        total_w = sum(weights)
        return np.argmax(sum(probs) / total_w, axis=1).tolist()
    else:
        # stacking
        feats = np.hstack([proba(m, texts) for m in models])
        meta = load(meta_path)
        return meta.predict(feats).tolist()


def train_meta(sub_models, val_texts, val_labels, strategy, meta_path):
    """
    Train and save a meta-model for stacking.

    Args:
        sub_models (List): list of model instances
        val_texts (List[str]): validation texts
        val_labels (List[int]): validation labels
        strategy (str): should be 'stacking' to trigger
        meta_path (str): where to save the trained meta-model
    """
    if strategy != 'stacking':
        return
    from lightgbm import LGBMClassifier

    # Collect level-1 features
    lvl1_feats = [proba(m, val_texts) for m in sub_models]
    X_meta = np.hstack(lvl1_feats)
    y_meta = val_labels

    meta_model = LGBMClassifier()
    meta_model.fit(X_meta, y_meta)

    return meta_model

