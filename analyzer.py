"""
analyzer.py — 파일 파싱 + 메타데이터 추출 + LLM 호출 핵심 로직
"""

import json
import io
import pandas as pd
from openai import AsyncOpenAI

from schemas import (
    AnalysisResponse, ColumnMapping, ColumnRole,
    DataMetadata, TaskType, VALID_ROLES_BY_TASK,
)
from prompt_builder import build_system_prompt, build_user_prompt


def _build_response_schema(task_type: TaskType) -> dict:
    """task_type에 맞는 role만 허용하는 JSON Schema 생성"""
    valid_roles = [r.value for r in VALID_ROLES_BY_TASK[task_type]]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "column_mapping_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "column_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "role":   {"type": "string", "enum": valid_roles},
                            },
                            "required": ["column", "role"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["column_mappings"],
                "additionalProperties": False,
            },
        },
    }


def parse_file_content(file_content: bytes, filename: str) -> tuple[list[str], pd.DataFrame]:
    """
    CSV 또는 JSON 파일을 파싱해 컬럼명 목록과 전체 DataFrame을 반환합니다.

    지원 JSON 구조:
      1. records 배열:  [{col: val, ...}, ...]
      2. 열 기반 dict:  {col: [val, ...], ...}
      3. 단일 키 래핑:  {"samples": [{...}, ...]}  ← 자동 언래핑
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_content))

    elif ext == "json":
        raw = json.loads(file_content.decode("utf-8"))

        if isinstance(raw, dict):
            values = list(raw.values())
            if len(values) == 1 and isinstance(values[0], list):
                raw = values[0]

        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            df = pd.DataFrame(raw)
        else:
            raise ValueError(
                "JSON 형식 오류: records 배열([{...}]) 또는 열 기반 dict({col: [...]}) 형태여야 합니다."
            )
    else:
        raise ValueError(f"지원하지 않는 파일 형식: .{ext}  (CSV 또는 JSON만 허용)")

    # 전체 DataFrame 반환 (메타데이터 추출용)
    return df.columns.tolist(), df


# ── 양성 클래스 자동 판단 규칙 ────────────────────────────────────────────────
# 숫자형: 큰 값이 Positive (1 > 0, True > False)
# 문자열 known patterns: 아래에 정의된 것만 자동 판단, 나머지는 ambiguous 처리
_KNOWN_POSITIVE = {"1", "yes", "true", "positive", "pos", "spam", "malignant", "fraud", "1.0"}
_KNOWN_NEGATIVE = {"0", "no", "false", "negative", "neg", "ham", "benign", "normal", "0.0"}


def _detect_binary_classes(series: pd.Series) -> tuple[str | None, str | None, bool]:
    """
    Binary y_true 컬럼에서 양성/음성 클래스를 자동 추론합니다.

    Returns:
        (positive_class, negative_class, is_ambiguous)
        is_ambiguous=True 이면 사용자 확인이 필요합니다.
    """
    unique_vals = [str(v) for v in series.dropna().unique()]
    if len(unique_vals) != 2:
        return None, None, True  # 2개 값이 아니면 판단 불가

    a, b = unique_vals[0], unique_vals[1]
    a_lower, b_lower = a.lower(), b.lower()

    # 숫자형: 큰 값이 Positive
    try:
        fa, fb = float(a), float(b)
        if fa > fb:
            return a, b, False
        else:
            return b, a, False
    except ValueError:
        pass

    # Known pattern 매칭
    if a_lower in _KNOWN_POSITIVE and b_lower in _KNOWN_NEGATIVE:
        return a, b, False
    if b_lower in _KNOWN_POSITIVE and a_lower in _KNOWN_NEGATIVE:
        return b, a, False

    # 판단 불가 → 알파벳순으로 첫 번째를 Positive로 임시 지정 후 ambiguous 표시
    pos, neg = sorted([a, b])
    return pos, neg, True


def extract_metadata(
    task_type: TaskType,
    df: pd.DataFrame,
    sample_df: pd.DataFrame,
    column_mappings: list[ColumnMapping],
) -> DataMetadata:
    """
    확정된 컬럼 매핑을 기반으로 메타데이터를 추출합니다.

    - 클래스 감지: sample_df (30행) 기준 → 속도 우선
    - 분포 계산:   df (전체)     기준 → 정확도 우선

    - Binary:     양성/음성 클래스 자동 판단
    - Multiclass: y_true 고유 클래스 목록 + 분포
    - Multilabel: true_labels 파싱 후 고유 레이블 목록 + 분포
    """
    role_to_col: dict[str, str] = {m.role.value: m.column for m in column_mappings}

    # ── Binary ────────────────────────────────────────────────────────────────
    if task_type == TaskType.binary:
        y_true_col = role_to_col.get(ColumnRole.y_true.value)
        if y_true_col and y_true_col in df.columns:
            # 클래스 감지: 샘플 30행으로
            pos, neg, ambiguous = _detect_binary_classes(sample_df[y_true_col])
            # 분포: 전체 df로
            distribution = df[y_true_col].value_counts().to_dict()
            distribution = {str(k): int(v) for k, v in distribution.items()}
            return DataMetadata(
                positive_class=pos,
                negative_class=neg,
                positive_class_ambiguous=ambiguous,
                class_distribution=distribution,
            )

    # ── Multiclass ────────────────────────────────────────────────────────────
    elif task_type == TaskType.multiclass:
        y_true_col = role_to_col.get(ColumnRole.y_true.value)
        if y_true_col and y_true_col in df.columns:
            # 클래스 감지: 샘플 30행의 unique 값으로
            classes = sorted({str(v) for v in sample_df[y_true_col].dropna().unique()})
            # 분포: 전체 df로
            distribution = df[y_true_col].value_counts().to_dict()
            distribution = {str(k): int(v) for k, v in distribution.items()}
            return DataMetadata(
                detected_classes=classes,
                class_distribution=distribution,
            )

    # ── Multilabel ────────────────────────────────────────────────────────────
    elif task_type == TaskType.multilabel:
        true_col = role_to_col.get(ColumnRole.true_labels.value)
        if true_col and true_col in df.columns:
            # 클래스 감지: 샘플 30행의 레이블 파싱으로
            sample_labels: set[str] = set()
            for cell in sample_df[true_col].dropna():
                for label in str(cell).split("|"):
                    label = label.strip()
                    if label:
                        sample_labels.add(label)
            labels = sorted(sample_labels)
            # 분포: 전체 df로
            label_counts: dict[str, int] = {}
            for cell in df[true_col].dropna():
                for label in str(cell).split("|"):
                    label = label.strip()
                    if label:
                        label_counts[label] = label_counts.get(label, 0) + 1
            return DataMetadata(
                detected_labels=labels,
                class_distribution=label_counts,
            )

    return DataMetadata()


async def analyze_columns_with_llm(
    client: AsyncOpenAI,
    task_type: TaskType,
    columns: list[str],
    df: pd.DataFrame,
) -> AnalysisResponse:
    """LLM으로 컬럼 역할을 자동 매핑하고, 데이터 메타데이터를 추출합니다."""
    # 30행 샘플: LLM 컬럼 매핑 + 클래스 감지용
    sample_df = df.head(30)

    response = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": build_system_prompt(task_type)},
            {"role": "user",   "content": build_user_prompt(columns, sample_df)},
        ],
        response_format=_build_response_schema(task_type),
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content)
    column_mappings = [
        ColumnMapping(column=m["column"], role=ColumnRole(m["role"]))
        for m in data["column_mappings"]
    ]

    # 클래스 감지: sample_df(30행) / 분포 계산: 전체 df
    metadata = extract_metadata(task_type, df, sample_df, column_mappings)

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata,
    )
