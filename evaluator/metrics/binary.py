import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
    matthews_corrcoef
)

def _get_true_pred(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('true_class')
    pred_col = mapping_dict.get('predicted_class')
    if not true_col or not pred_col:
        raise ValueError("true_class 또는 predicted_class 컬럼 매핑이 필요합니다.")
    return df[true_col], df[pred_col]

def _get_true_score(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('true_class')
    score_col = mapping_dict.get('score')
    if not true_col or not score_col:
        raise ValueError("true_class 또는 score(확률) 컬럼 매핑이 필요합니다.")
    return df[true_col], df[score_col]

def _binarize_true_labels(y_true):
    """문자열 등 비숫자형 라벨을 0, 1로 안전하게 변환"""
    classes = np.sort(np.unique(y_true))
    if len(classes) <= 2:
        pos_label = classes[-1]  # 정렬 시 마지막 값을 positive(1)로 간주 (e.g. 'Yes', 1)
        return (y_true == pos_label).astype(int)
    return y_true

def calculate_specificity(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC7: Specificity (True Negative Rate)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return 0.0

def calculate_fpr(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC8: False Positive Rate (FPR)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return 0.0

def calculate_auroc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC9: Area Under the Receiver Operating Characteristic Curve (AUROC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    y_true_bin = _binarize_true_labels(y_true)
    return float(roc_auc_score(y_true_bin, y_score))

def calculate_auprc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC10: Area Under the Precision-Recall Curve (AUPRC)"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    y_true_bin = _binarize_true_labels(y_true)
    return float(average_precision_score(y_true_bin, y_score))

def calculate_log_loss(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC19: Log Loss"""
    y_true, y_score = _get_true_score(df, mapping_dict)
    y_true_bin = _binarize_true_labels(y_true)
    return float(log_loss(y_true_bin, y_score))

def calculate_mcc(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC20: Matthews Correlation Coefficient (MCC)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(matthews_corrcoef(y_true, y_pred))

