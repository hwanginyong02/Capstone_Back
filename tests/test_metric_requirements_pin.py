"""METRIC_REQUIREMENTS·VALID_ROLES_BY_TASK 의 **교차 저장소 고정핀** — ISSUES.md A-03.

## 왜 이 파일이 필요한가

지표→필수컬럼 규칙은 네 곳에 있다(A-03). 5차 라운드는 "계약 테스트가 드리프트를 CI 에서
즉시 red 로 잡는다"고 기록했는데, **한쪽 방향만 참이었다.**

- 프론트의 계약 테스트(`Capstone_Front/src/data/metricColumnContract.test.ts`)는 프론트 표를
  **같은 파일 안의 손 사본** `BACKEND_METRIC_REQUIREMENTS` 와 비교한다. 프론트 표를 고치면
  깨지지만, **백엔드 표를 고쳐도 그 손 사본은 그대로**라 프론트 CI 는 초록이다.
- 백엔드에는 `test_metric_requirements.py` 가 있지만 그것은 표를 **실제 계산 함수**와
  대조한다. 표와 구현을 **함께** 고치면(새 지표 추가, 어떤 지표의 요구 역할 변경) 통과한다.
- 두 저장소의 CI 는 서로를 checkout 하지 않는다.

그래서 **표와 구현을 함께 바꾸는 백엔드 단독 변경은 양쪽 CI 를 모두 통과하고 배포된다.**
그 결과가 어디서 터지는지가 중요하다 — 5단계 '다음' 게이트는 프론트가 **로컬 표로**
판정하므로(`mappingValidity.ts`), 낡은 프론트 표는 백엔드가 받아줄 매핑을 화면에서
**하드 차단**한다. A-13 이 정확히 그 실패 형태였다.

## 이 테스트가 하는 일

표를 **명시적 기대값에 고정(pin)** 한다. 값을 비교하는 것 자체가 목적이 아니라,
**백엔드 표를 건드리면 반드시 이 파일도 건드리게 만드는 것**이 목적이다. 그때 실패
메시지가 함께 고쳐야 할 프론트 파일 두 개를 이름으로 알려준다.

사본을 하나 더 만드는 셈이지만, 이 사본은 **같은 저장소·같은 CI 안에 있어 사람의 기억에
의존하지 않는다.** 사본 4곳을 하나로 줄이는 일(백엔드 표를 API 로 내려보내 프론트가 소비)은
약 26개 파일과 두 저장소의 순서 있는 배포가 필요하고, 워크플로우 화면 셋과 **랜딩 페이지**를
동기 렌더에서 fetch 의존으로 바꾸며(Render free 인스턴스), 게다가 두 소비처는 애초에
await 할 수 없다(`reportShowcase.ts` 는 모듈 로드 시점에 계산하고, zustand persist 의
`migrate` 는 동기 함수다). 남는 사본도 최대 둘뿐이라(SPEC.md 는 산문이고 `METRICS` 는
성적서가 인쇄하는 이름·산식을 갖고 있다) 비용 대비 값이 맞지 않는다.
그 판단 근거는 `docs/ARCHITECTURE.md` §8.5 에 있다.
"""

from app.core.schemas import (
    METRIC_REQUIREMENTS,
    PREDICTION_ROLES_BY_TASK,
    VALID_ROLES_BY_TASK,
    ColumnRole,
    TaskType,
)

# 프론트에서 함께 고쳐야 하는 파일들. 실패 메시지에 그대로 싣는다.
_FRONTEND_MIRRORS = """
이 표를 바꾸려면 Capstone_Front 의 아래 두 곳을 **같은 PR 로** 함께 고쳐야 한다.
어느 한쪽만 고치면 5단계 '다음' 게이트가 프론트 표로 판정하므로, 백엔드가 받아줄
매핑을 화면이 하드 차단한다(ISSUES.md A-13 의 실패 형태).

  1. src/data/evaluationData.ts
       - REQUIRED_COLUMNS_BY_METRIC  (프론트 어휘: id/y_true/y_pred/score/prob_class_*/prob_label_*)
       - METRICS[].supportedTaskTypes (지표 집합이 바뀔 때)
  2. src/data/metricColumnContract.test.ts
       - BACKEND_METRIC_REQUIREMENTS (이 표의 손 사본)
       - BACKEND_VALID_ROLES         (VALID_ROLES_BY_TASK 의 손 사본)

역할 이름은 두 저장소에서 다르다 — 대조는 반드시
Capstone_Front/src/lib/mapping/translateRoleToBackend.ts 를 통과시켜라
(id↔sample_id, score↔score_positive, prob_class_*↔prob_per_class,
 prob_label_*↔score_per_label, y_true↔true_labels, y_pred↔pred_labels).

SPEC.md(§1~§3 허용 지표·요구 컬럼)도 정본이므로 함께 고친다.
"""

_Y = ColumnRole.y_true
_P = ColumnRole.y_pred
_S = ColumnRole.score_positive
_TL = ColumnRole.true_labels
_PL = ColumnRole.pred_labels

# 고정 기대값. 이 리터럴을 바꿀 때 위 두 프론트 파일을 함께 바꾼다.
EXPECTED_METRIC_REQUIREMENTS: dict[TaskType, dict[str, set[ColumnRole]]] = {
    TaskType.binary: {
        "M1": {_Y, _P}, "M2": {_Y, _P}, "M3": {_Y, _P}, "M4": {_Y, _P},
        "M5": {_Y, _P}, "M6": {_Y, _P}, "M7": {_Y, _P}, "M8": {_Y, _P},
        "M9": {_Y, _S}, "M10": {_Y, _S}, "M19": {_Y, _S},
        "M20": {_Y, _P}, "M21": {_Y, _P}, "M22": {_Y, _P},
        "M23": {_Y},
    },
    TaskType.multiclass: {
        "M1": {_Y, _P}, "M2": {_Y, _P}, "M3": {_Y, _P}, "M4": {_Y, _P},
        "M5": {_Y, _P}, "M6": {_Y, _P},
        "M11": {_Y, _P}, "M12": {_Y, _P}, "M13": {_Y, _P}, "M14": {_Y, _P},
        "M21": {_Y, _P}, "M22": {_Y, _P},
        "M23": {_Y},
    },
    TaskType.multilabel: {
        "M2": {_TL, _PL}, "M3": {_TL, _PL}, "M4": {_TL, _PL}, "M5": {_TL, _PL},
        "M15": {_TL, _PL}, "M16": {_TL, _PL}, "M17": {_TL, _PL}, "M18": {_TL, _PL},
        "M21": {_TL, _PL}, "M22": {_TL, _PL},
        "M23": {_TL},
    },
}

EXPECTED_VALID_ROLES: dict[TaskType, list[ColumnRole]] = {
    TaskType.binary: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.score_positive, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multiclass: [
        ColumnRole.sample_id, ColumnRole.y_true, ColumnRole.y_pred,
        ColumnRole.prob_per_class, ColumnRole.latency, ColumnRole.ignore,
    ],
    TaskType.multilabel: [
        ColumnRole.sample_id, ColumnRole.true_labels, ColumnRole.pred_labels,
        ColumnRole.score_per_label, ColumnRole.latency, ColumnRole.ignore,
    ],
}


def _describe(table: dict[str, set[ColumnRole]]) -> str:
    return "\n".join(
        f"    {metric_id}: {sorted(r.value for r in roles)}"
        for metric_id, roles in sorted(table.items())
    )


def test_metric_requirements_is_pinned():
    """표가 고정 기대값과 정확히 같아야 한다(지표 집합·역할 집합 모두)."""
    assert set(METRIC_REQUIREMENTS) == set(EXPECTED_METRIC_REQUIREMENTS), (
        "METRIC_REQUIREMENTS 의 task_type 집합이 바뀌었다." + _FRONTEND_MIRRORS
    )

    for task in EXPECTED_METRIC_REQUIREMENTS:
        actual = METRIC_REQUIREMENTS[task]
        expected = EXPECTED_METRIC_REQUIREMENTS[task]

        # 지표 집합의 차이를 먼저 말한다 — 역할 diff 보다 원인이 분명하다.
        added = sorted(set(actual) - set(expected))
        removed = sorted(set(expected) - set(actual))
        assert not added and not removed, (
            f"{task.value}: 지표 집합이 바뀌었다 — 추가 {added}, 삭제 {removed}.\n"
            f"  현재 표:\n{_describe(actual)}" + _FRONTEND_MIRRORS
        )

        changed = {
            metric_id: (
                sorted(r.value for r in expected[metric_id]),
                sorted(r.value for r in actual[metric_id]),
            )
            for metric_id in expected
            if actual[metric_id] != expected[metric_id]
        }
        assert not changed, (
            f"{task.value}: 요구 역할이 바뀌었다 (지표: 고정값 → 현재값)\n"
            + "\n".join(f"    {m}: {was} → {now}" for m, (was, now) in sorted(changed.items()))
            + _FRONTEND_MIRRORS
        )


def test_valid_roles_by_task_is_pinned():
    """매핑 화면 드롭다운의 근거 목록도 함께 고정한다.

    이 목록을 벗어난 역할은 `translateRoleToBackend` 에서 조용히 `ignore` 로 강등되어
    해당 컬럼이 평가에서 빠진다 — 화면에는 아무 오류도 뜨지 않는다.
    """
    assert {t: list(r) for t, r in VALID_ROLES_BY_TASK.items()} == EXPECTED_VALID_ROLES, (
        "VALID_ROLES_BY_TASK 가 바뀌었다." + _FRONTEND_MIRRORS
    )


def test_requirement_table_never_demands_sample_id():
    """`sample_id` 는 어떤 행에도 나타나서는 안 된다 (ISSUES.md A-13).

    SPEC §0 이 '선택(권장)'으로 규정한다. 종전에 프론트 요구표 43개 항목이 전부 "id" 로
    시작해, id 컬럼이 없는 데이터셋은 매핑 화면에서 하드 차단돼 백엔드까지 오지도 못했다.
    """
    offenders = [
        f"{task.value} {metric_id}"
        for task, table in METRIC_REQUIREMENTS.items()
        for metric_id, roles in table.items()
        if ColumnRole.sample_id in roles
    ]
    assert offenders == [], (
        f"요구표가 sample_id 를 강제한다: {offenders}. "
        "sample_id 는 선택 컬럼이며 중복 검사에만 쓰인다(ISSUES.md A-13)."
    )


def test_prediction_alternatives_cover_every_task():
    """확률→예측 대체 규칙은 세 task 전부에 있어야 한다 (ISSUES.md A-01·A-02, ★결정 1).

    이 표가 비면 확률만 있는 데이터셋이 다시 구조적으로 차단된다. 요구표만 대조하는
    검사는 이 택일 규칙을 보지 못하므로(요구표의 값은 set 이라 AND 만 표현한다)
    여기서 따로 고정한다. 프론트 거울은 `evaluationData.ts` 의
    `PREDICTION_ROLE_ALTERNATIVES` 다.
    """
    assert set(PREDICTION_ROLES_BY_TASK) == set(TaskType)
    for task, (primary, alternatives) in PREDICTION_ROLES_BY_TASK.items():
        assert alternatives, f"{task.value}: 예측 역할 {primary.value} 의 대체 역할이 비어 있다"
        assert primary not in alternatives
