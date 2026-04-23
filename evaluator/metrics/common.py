import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    fbeta_score,
    confusion_matrix,
    classification_report
)

def _get_true_pred(df: pd.DataFrame, mapping_dict: dict):
    true_col = mapping_dict.get('true_class')
    pred_col = mapping_dict.get('predicted_class')
    if not true_col or not pred_col:
        raise ValueError("true_class 또는 predicted_class 컬럼 매핑이 필요합니다.")
    return df[true_col], df[pred_col]

def calculate_accuracy(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC1: Accuracy"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(accuracy_score(y_true, y_pred))

def calculate_precision(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC2: Precision (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    # 클래스 이름이 문자열일 경우를 대비해 가장 안전한 macro average 사용
    return float(precision_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_recall(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC3: Recall (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(recall_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_f1_score(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC4: F1 Score (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))

def calculate_fbeta_score(df: pd.DataFrame, mapping_dict: dict, beta: float = 1.0) -> float:
    """TC5: F-beta Score (Macro Average)"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    # 차후 프론트에서 beta 값을 넘겨주면 활용 가능하도록 설계
    return float(fbeta_score(y_true, y_pred, beta=beta, average='macro', zero_division=0))

def calculate_kl_divergence(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC6: KL Divergence"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # 확률 분포 계산 (normalize=True)
    p_counts = pd.Series(y_true).value_counts(normalize=True)
    q_counts = pd.Series(y_pred).value_counts(normalize=True)
    
    # 0으로 나누는 것을 방지하기 위해 아주 작은 값(epsilon) 추가
    epsilon = 1e-9
    p = np.array([p_counts.get(c, epsilon) for c in classes])
    q = np.array([q_counts.get(c, epsilon) for c in classes])
    
    kl_div = np.sum(p * np.log(p / q))
    return float(kl_div)

def calculate_confusion_matrix(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """TC21: Confusion Matrix"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "labels": [str(l) for l in labels],
        "matrix": cm.tolist() # JSON 직렬화를 위해 리스트로 변환
    }

def calculate_class_metrics(df: pd.DataFrame, mapping_dict: dict) -> Dict[str, Any]:
    """TC22: Class별 Metric"""
    y_true, y_pred = _get_true_pred(df, mapping_dict)
    # output_dict=True를 통해 JSON 형태로 바로 내려주기 좋음
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report

def calculate_imbalance_ratio(df: pd.DataFrame, mapping_dict: dict) -> float:
    """TC23: Imbalance Ratio (가장 많은 클래스 / 가장 적은 클래스 비율)"""
    true_col = mapping_dict.get('true_class')
    if not true_col:
        raise ValueError("true_class 컬럼 매핑이 필요합니다.")
    
    counts = df[true_col].value_counts()
    if len(counts) == 0:
        return 0.0
    
    majority = counts.max()
    minority = counts.min()
    
    if minority == 0:
        return float('inf')
        
    return float(majority / minority)

