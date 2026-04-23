import pandas as pd
from typing import Dict, Any, List

from .metrics import common, binary, multiclass, multilabel
from .preprocessor import preprocess_data

# Task Type 별로 허용되는 TC 정의 (안전 장치)
VALID_TCS_BY_TASK = {
    "binary": {"TC1", "TC2", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9", "TC10", "TC19", "TC20", "TC21", "TC22", "TC23"},
    "multiclass": {"TC1", "TC2", "TC3", "TC4", "TC5", "TC6", "TC11", "TC12", "TC13", "TC14", "TC21", "TC22", "TC23"},
    "multilabel": {"TC1", "TC2", "TC3", "TC4", "TC5", "TC15", "TC16", "TC17", "TC18", "TC21", "TC22", "TC23"}
}

# TC ID 와 실제 계산 함수 매핑 (Registry)
METRIC_REGISTRY = {
    "TC1": common.calculate_accuracy,
    "TC2": common.calculate_precision,
    "TC3": common.calculate_recall,
    "TC4": common.calculate_f1_score,
    "TC5": common.calculate_fbeta_score,
    "TC6": common.calculate_kl_divergence,
    "TC7": binary.calculate_specificity,
    "TC8": binary.calculate_fpr,
    "TC9": binary.calculate_auroc,
    "TC10": binary.calculate_auprc,
    "TC11": multiclass.calculate_macro_average,
    "TC12": multiclass.calculate_micro_average,
    "TC13": multiclass.calculate_weighted_average,
    "TC14": multiclass.calculate_distribution_diff_mc,
    "TC15": multilabel.calculate_hamming_loss,
    "TC16": multilabel.calculate_exact_match_ratio,
    "TC17": multilabel.calculate_jaccard_index,
    "TC18": multilabel.calculate_distribution_diff_ml,
    "TC19": binary.calculate_log_loss,
    "TC20": binary.calculate_mcc,
    "TC21": common.calculate_confusion_matrix,
    "TC22": common.calculate_class_metrics,
    "TC23": common.calculate_imbalance_ratio,
}


def evaluate(df: pd.DataFrame, mappings: List[Dict[str, str]], task_type: str, selected_tcs: List[str]) -> Dict[str, Any]:
    """
    메인 평가 엔진 진입점.
    Task Type에 검증된 TC들만 필터링하여 동적으로 실행합니다.
    
    Args:
        df: 전처리/검증이 완료된 DataFrame
        mappings: 프론트에서 확정하여 전달한 역할 매핑 리스트 [{"column": "col_A", "role": "true_class"}, ...]
        task_type: "binary" | "multiclass" | "multilabel"
        selected_tcs: 클라이언트가 요청한 평가 지표 리스트 ["TC1", "TC7"]
        
    Returns:
        평가 결과 (최종 리포트 딕셔너리 형태)
    """
    results = {}
    valid_tcs = VALID_TCS_BY_TASK.get(task_type, set())
    
    # ── [전처리 단계 추가] ──
    try:
        df, pre_logs = preprocess_data(df, mappings, task_type)
        results["_metadata"] = pre_logs
    except Exception as e:
        # 전처리 단계에서 에러(예: 필수 컬럼 누락, 확률 범위 초과)가 나면 즉시 반환
        return {"error": str(e)}

    # mappings 리스트를 딕셔너리로 변환 ({"true_class": "col_A", ...} 형태)하여 하위 함수들이 편하게 쓰도록 변경
    mapping_dict = {item['role']: item['column'] for item in mappings}
    
    for tc_id in selected_tcs:
        if tc_id not in valid_tcs:
            results[tc_id] = {"error": f"{task_type}에서는 지원하지 않는 지표입니다."}
            continue
            
        if tc_id in METRIC_REGISTRY:
            func = METRIC_REGISTRY[tc_id]
            try:
                # 매핑된 계산 함수 실행
                results[tc_id] = func(df, mapping_dict)
            except Exception as e:
                # 에러가 나더라도 다른 지표 계산에 영향을 주지 않도록 격리
                results[tc_id] = {"error": str(e)}
        else:
            results[tc_id] = {"error": "구현되지 않은 지표입니다."}
            
    return results
