"""평가 S1 — 새 결과서 3건의 판 4 요청을 뜰 입력(평가·검증 응답, 스토어 상태)을 만든다. 사전 등록 v1 §2.

개발 3건과 입력 상태를 맞춘다(고객 입력은 모의 — 대응 개발 결과서의 항목을 그대로 본뜨고, 데이터셋에 맞는 이름·용도·분량만 바꾼다):
- pima_strong  ← bc_strong   (프론트 edition3/newRun/inputs/bc_strong.state.json 을 바탕으로)
- segment_weak ← digits_weak (c_view_fixtures.json FIXTURES[digits_weak] 와 같은 칸을 비우고, 1층 필수 칸만 채움 — j0 WEAK_FILL 과 같음)
- scene_strong ← yeast_strong(프론트 edition3/newRun/inputs/yeast_strong.state.json 을 바탕으로)

목표값은 사전 등록 v1 §2-3 공식으로 이 도구가 계산한다(달성값은 평가 응답에서 읽음).
평가 응답의 environment 는 j0 와 같이 bc_strong 계약 입력 값으로 덮는다(실행마다 시각·버전이 흔들리지 않게).

쓰는 법(.env 없는 백엔드 사본 폴더에서): <백엔드 venv python> <이 파일> <Capstone 폴더> <데이터 폴더(e1 출력)> <입력 출력 폴더>
"""
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
assert not Path(".env").exists(), ".env 가 있는 폴더에서 돌리지 않는다"
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ROOT, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NEW_RUN = ROOT / "Capstone_Front/src/test/edition3/newRun/inputs"
NEW_KEYS = ROOT / "Capstone_Front/src/test/edition4/evaluateNewKeys"
C_FIXTURES = ROOT / "P4_ROUND/실데이터/판3발급본/c_view_fixtures.json"
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
dump = lambda v: json.dumps(v, ensure_ascii=False, indent=1) + "\n"  # noqa: E731
entries = lambda o: {"entries": [{"key": k, "value": v} for k, v in o.items()]}  # noqa: E731
ENV = load(NEW_KEYS / "bc_strong.evaluate.json")["environment"]
TEMPLATE_JOBS = load(NEW_RUN / "bc_strong.jobs.json")
ROLE = {"sample_id": "id", "y_true": "y_true", "true_labels": "y_true", "y_pred": "y_pred", "pred_labels": "y_pred",
        "score_positive": "score", "prob_per_class": "prob_class_*", "score_per_label": "prob_label_*"}
LOWER_BETTER = {"M6", "M15", "M18", "M19", "M23"}


def target(metric: str, v: float, side: str) -> str:
    """사전 등록 v1 §2-3. side='pass'(넘게) | 'fail'(못 미치게). 소수 둘째 자리 글로 돌려준다."""
    lo, hi = math.floor(round(v * 100, 9)) / 100, math.ceil(round(v * 100, 9)) / 100
    if metric in LOWER_BETTER:
        t = hi + 0.02 if side == "pass" else lo - 0.02
    else:
        t = lo - 0.02 if side == "pass" else hi + 0.02
    return f"{t:.2f}"


client = TestClient(app)
evals = {}
for tag in ["pima_strong", "segment_weak", "scene_strong"]:
    csv, payload = DATA / f"{tag}.csv", load(DATA / f"{tag}.payload.json")
    files = {"file": (csv.name, csv.read_bytes(), "text/csv")}
    form = {"data": json.dumps(payload)}
    ev = client.post("/api/evaluate", files=files, data=form)
    va = client.post("/api/validate-data", files=files, data=form)
    assert ev.status_code == 200 and va.status_code == 200, (tag, ev.status_code, va.status_code, ev.text[:300], va.text[:300])
    evaluate = {**ev.json(), "environment": ENV}
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "payload.json").write_text(dump(payload), encoding="utf-8")
    (d / "evaluate.json").write_text(dump(evaluate), encoding="utf-8")
    (d / "validate.json").write_text(dump(va.json()), encoding="utf-8")
    (d / "data.csv").write_bytes(csv.read_bytes())
    evals[tag] = (payload, evaluate, csv.read_bytes())

val = lambda tag, m: float(evals[tag][1]["results"]["success_metrics"][m])  # noqa: E731


def columns(payload):
    return [{"originalName": m["column"], "sampleValues": m["sample_values"], "inferredRole": ROLE[m["role"]],
             "confirmedRole": ROLE[m["role"]], "modified": False, "warnings": []} for m in payload["column_mappings"]]


def details(targets):
    return {i: {"id": i, "name": i, "description": "", "targetValue": v, "beta": "", "positiveClass": "", "completed": True}
            for i, v in targets.items()}


def finish(tag, base_state, workflow, provenance):
    payload, _, csv_bytes = evals[tag]
    workflow.update({
        "selectedMetricIds": payload["selected_metric_ids"],
        "columnMapping": columns(payload),
        "metadata": payload["metadata"],
        "decisionThreshold": payload["decision_threshold"],
    })
    assert list(workflow) == list(base_state["workflow"]), f"{tag}: 스토어 상태 키가 바탕 입력과 다르다"
    file = {**base_state["file"], "name": f"{tag}.csv", "bytes": len(csv_bytes)}
    state = {"$provenance": provenance, "file": file, "workflow": workflow}
    (OUT / tag / "state.json").write_text(dump(state), encoding="utf-8")
    jobs = json.loads(json.dumps(TEMPLATE_JOBS).replace("job-s0b-bc_strong", f"job-e1-{tag}"))
    jobs["$provenance"] = "bc_strong.jobs.json 사본(작업 번호만 바꿈) — 요청 본문을 뜨는 데는 쓰이지 않는다"
    (OUT / tag / "jobs.json").write_text(dump(jobs), encoding="utf-8")
    return state


targets_out = {}

# ---- pima_strong ← bc_strong (입력 완전, 위험 방향 FN, 핵심 M3)
bc = load(NEW_RUN / "bc_strong.state.json")
w = copy.deepcopy(bc["workflow"])
t = {m: target(m, val("pima_strong", m), "pass") for m in ["M1", "M3", "M4", "M9", "M19"]}
targets_out["pima_strong"] = t
w["basicInfo"].update({
    "reportPurposeNote": "의료기기 소프트웨어 사전 상담 자료로 제출",
    "versionName": "v1.4",
    "modelName": "DiabetesScreen-LR",
    "modelPurpose": "검진 기록(임신 횟수·혈당·혈압·피부 두께·인슐린·체질량지수·가족력 지수·나이)으로 당뇨 양성 여부를 가려, 정밀 검사 안내 대상의 우선순위를 정하는 데 쓴다.",
    "modelCategory": "의료 검진 보조 분류",
})
w["metricDetails"] = details(t)
w["classLabelDescriptions"] = {"tested_positive": "당뇨 양성 판정", "tested_negative": "당뇨 음성 판정"}
w["composerMetrics"] = {"coreMetrics": {"choices": ["M3"]}}
w["composerPerspective"] = {
    "criticalErrorType": {"text": "Missed (FN)", "note": "양성을 놓치면 정밀 검사와 치료가 늦어진다. 잘못 잡은 경우는 재검사로 걸러진다."},
    "usageMode": {"text": "Batch", "note": "하루 검진분을 밤에 한 번에 돌린다."},
}
w["composerTrainingData"] = {
    "trainingDatasetName": {"text": "Pima Indians Diabetes 학습 분할"},
    "trainingDataSource": {"text": "미국 국립당뇨병·소화기·신장질환연구소 검진 기록(공개 데이터)"},
    "trainingVolume": entries({"Training": "537", "Validation": "0"}),
    "trainingClassVolume": entries({"tested_positive": "187", "tested_negative": "350"}),
    "channelEffects": {"choices": ["None known"]},
    "imbalanceHandling": {"choices": ["Did nothing"], "note": "클래스 비율(약 35:65)이 크게 기울지 않아 따로 맞추지 않았다."},
    "preprocessingMatch": {"text": "Same"},
}
w["composerTestData"] = {
    "testDataSource": {"text": "같은 데이터의 30% 층화 분할(학습에 쓰지 않음)"},
    "testSourceRelation": {"text": "Same source", "note": "같은 기관·같은 검사 절차에서 나온 기록이다."},
}
w["composerGroundTruth"] = {
    "labelAuthor": {"text": "Domain experts", "note": "WHO 기준 경구 포도당 부하 검사로 확인한 진단 결과"},
    "labelReview": {"text": "Several reviewers"},
    "labelReviewerCount": {"text": "2"},
    "labelAgreementKind": {"text": "Cohen's kappa"},
    "labelAgreementValue": {"text": "0.90"},
}
w["composerModelEnv"] = copy.deepcopy(bc["workflow"]["composerModelEnv"])  # 로지스틱 회귀(입력 표준화), C 0.5·max_iter 2000·lbfgs
assert w["composerModelEnv"]["algorithm"]["text"] == "로지스틱 회귀(입력 표준화)"
finish("pima_strong", bc, w, "평가 S1 e2_inputs.py — bc_strong.state.json 을 바탕으로 한 모의 고객 입력 + pima_strong.payload.json")

# ---- segment_weak ← digits_weak (부실 입력, 핵심 지표 지정 안 함 — 기본 M1·M4)
dg = next(f for f in load(C_FIXTURES) if f["tag"] == "digits_weak")
basic = {**bc["workflow"]["basicInfo"], "projectName": "", "projectAgency": "", "projectNumber": "", **dg["basic"],
         "taskType": "multiclass", "modelName": "SegmentReader-Tree", "versionName": "v0.1",
         # 1층 필수 칸만 채운다(j0_states.py WEAK_FILL[digits_weak] 와 같은 두 칸)
         "modelPurpose": "야외 사진의 작은 영역(3×3 화소)을 일곱 종류(벽돌면·시멘트·나뭇잎·창문·길·잔디·하늘)로 나눠 영상 분할에 넘긴다.",
         "modelCategory": "영상 영역 분류"}
seg_targets = dict(dg["targets"])  # M1 0.9 · M4 0.85 (digits_weak 와 같음)
targets_out["segment_weak"] = seg_targets
w = {
    "basicInfo": basic,
    "taskType": "multiclass",
    "selectedMetricIds": None,
    "metricDetails": details(seg_targets),
    "columnMapping": None,
    "metadata": None,
    "decisionThreshold": None,
    "classLabelDescriptions": {},
    "trainingExampleFiles": [],
    "trainingUnsuitableExampleFiles": [],
    "composerCards": dict(dg["cards"]),
    "composerMetrics": {},
    "composerPerspective": copy.deepcopy(dg["perspective"]),
    "composerTrainingData": {
        "trainingDatasetName": {"text": "영상 영역 학습 분할"},
        "trainingDataSource": {"text": "UCI"},
        "trainingVolume": entries({"Training": "1617", "Validation": "0"}),
        "trainingClassVolume": {"unknown": True},
        "channelEffects": {"choices": ["Other (free text)"], "otherText": "촬영 카메라가 두 종류"},
        "imbalanceHandling": {"choices": ["Did nothing"]},
    },
    "composerTestData": {},
    "composerGroundTruth": copy.deepcopy(dg["truth"]),
    "composerModelEnv": {},
}
assert dg["cards"] == {"trainingData": True, "testData": False, "groundTruth": True, "modelEnv": False}
w = {k: w[k] for k in bc["workflow"]}  # 키 순서를 바탕과 같게
finish("segment_weak", bc, w, "평가 S1 e2_inputs.py — FIXTURES[digits_weak] 와 같은 칸을 비운 모의 고객 입력 + segment_weak.payload.json + 1층 필수 칸 채움")

# ---- scene_strong ← yeast_strong (같은 입력 항목, 핵심 M17·M15)
ys = load(NEW_RUN / "yeast_strong.state.json")
w = copy.deepcopy(ys["workflow"])
t = {"M17": target("M17", val("scene_strong", "M17"), "pass"), "M15": target("M15", val("scene_strong", "M15"), "pass"),
     "M4": target("M4", val("scene_strong", "M4"), "fail")}
targets_out["scene_strong"] = t
w["basicInfo"].update({
    "versionName": "v2.0",
    "modelName": "SceneTagger-OvR",
    "modelPurpose": "풍경 사진의 색·질감 특징으로 장면 범주(6개, 여러 개 가능)를 붙여 사진 검색 색인을 만든다.",
    "modelCategory": "영상 다중 레이블 분류",
})
w["metricDetails"] = details(t)
w["classLabelDescriptions"] = {"Beach": "해변", "FallFoliage": "단풍", "Urban": "도시"}
w["composerMetrics"] = {"coreMetrics": {"choices": ["M17", "M15"]},
                        "metricRationale": {"text": "Degree of overlap", "note": "한 사진의 장면 범주를 일부라도 맞히면 검색 색인에 쓸 수 있다."}}
w["composerPerspective"] = {"classPriority": {"text": "All equal"},
                            "usageMode": {"text": "Batch", "note": "주 1회 새 사진 목록을 한꺼번에 돌린다."}}
w["composerTrainingData"] = {
    "trainingDatasetName": {"text": "Mulan scene 학습 분할"},
    "trainingDataSource": {"text": "Boutell 외(2004) 공개 데이터"},
    "trainingVolume": entries({"Training": "1211", "Validation": "0"}),
    "channelEffects": {"choices": ["None known"]},
    "imbalanceHandling": {"choices": ["Did nothing"], "note": "레이블별 빈도 차이가 크지 않아(최대 2배 안쪽) 그대로 학습했다."},
    "preprocessingMatch": {"text": "Same"},
}
w["composerTestData"] = {"testDataSource": {"text": "Mulan 표준 시험 분할(1,196건)"}, "testSourceRelation": {"text": "Same source"}}
w["composerGroundTruth"] = {
    "labelAuthor": {"text": "Domain experts", "note": "사람 판독자가 붙인 장면 범주 주석"},
    "labelReview": {"text": "Several reviewers"},
    "labelReviewerCount": {"text": "3"},
    "labelAgreementKind": {"text": "Not measured"},
}
w["composerModelEnv"] = copy.deepcopy(ys["workflow"]["composerModelEnv"])  # 레이블별 로지스틱 회귀(One-vs-Rest), C 0.3·max_iter 3000
assert w["composerModelEnv"]["algorithm"]["text"] == "레이블별 로지스틱 회귀(One-vs-Rest)"
finish("scene_strong", ys, w, "평가 S1 e2_inputs.py — yeast_strong.state.json 을 바탕으로 한 모의 고객 입력 + scene_strong.payload.json")

(DATA / "목표값.json").write_text(dump(targets_out), encoding="utf-8")
for tag, tg in targets_out.items():
    print(tag, {m: (round(val(tag, m), 4), tg[m]) for m in tg})
