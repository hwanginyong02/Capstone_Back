"""평가 S1 — 새 결과서 3건(pima_strong·segment_weak·scene_strong)의 예측 CSV 와 평가 응답(백엔드 엔진, 키 없음).

개발 3건을 만든 P4_ROUND/도구/c_make_data.py 와 같은 꼴(분할·모델·열 이름 규칙)이다. 사전 등록 v1 §2.
쓰는 법(Capstone_Back 폴더에서):
  venv/bin/python <이 파일> <openml data_home> <mulan scene 폴더> <출력 폴더> [--segment-depth N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path.cwd()))
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("openml_home")
ap.add_argument("scene_dir")
ap.add_argument("out")
ap.add_argument("--segment-depth", type=int, default=4)
ap.add_argument("--only", default="")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
client = TestClient(app)
ONLY = set(filter(None, args.only.split(",")))


def evaluate(name, df, task, mappings, metrics, metadata, threshold=None):
    csv = OUT / f"{name}.csv"
    df.to_csv(csv, index=False)
    payload = {
        "task_type": task,
        "column_mappings": [
            {"column": c, "role": r, "sample_values": [str(v) for v in df[c].head(3).tolist()]} for c, r in mappings
        ],
        "selected_metric_ids": metrics,
        "metadata": metadata,
        "beta": 1,
        "decision_threshold": threshold,
    }
    (OUT / f"{name}.payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    r = client.post("/api/evaluate", files={"file": (csv.name, csv.read_bytes(), "text/csv")},
                    data={"data": json.dumps(payload)})
    assert r.status_code == 200, (name, r.status_code, r.text[:500])
    (OUT / f"{name}.evaluate.json").write_text(json.dumps(r.json(), ensure_ascii=False, indent=1))
    sm = r.json()["results"]["success_metrics"]
    print(name, len(df), {k: (v.get("value") if isinstance(v, dict) else None) for k, v in sm.items()
                          if k not in ("confusion_matrix", "class_report", "M21", "M22")})


# ---- 이진: Pima Indians Diabetes (OpenML diabetes v1, id 37). 양성 클래스 = tested_positive(찾아내야 하는 쪽)
if not ONLY or "pima_strong" in ONLY:
    pm = fetch_openml(name="diabetes", version=1, as_frame=True, data_home=args.openml_home, parser="auto")
    X = pm.data.to_numpy(float)
    y = (pm.target.astype(str) == "tested_positive").astype(int).to_numpy()  # 1 = 양성
    names = np.array(["tested_negative", "tested_positive"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    meta = {"positive_class": "tested_positive", "negative_class": "tested_negative", "positive_class_ambiguous": False,
            "detected_classes": ["tested_negative", "tested_positive"], "detected_labels": [],
            "class_distribution": {k: int(v) for k, v in pd.Series(names[yte]).value_counts().items()}}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model.fit(Xtr, ytr)
    p_pos = model.predict_proba(Xte)[:, 1]
    df = pd.DataFrame({"sample_id": [f"PM{i:04d}" for i in range(len(yte))], "outcome": names[yte],
                       "score_tested_positive": np.round(p_pos, 4)})  # 강한 고객 = 확률만 올림(임계값 0.5 로 예측 파생)
    maps = [("sample_id", "sample_id"), ("outcome", "y_true"), ("score_tested_positive", "score_positive")]
    evaluate("pima_strong", df, "binary", maps, ["M1", "M2", "M3", "M4", "M9", "M10", "M19", "M23"], meta, 0.5)
    print("  split", len(ytr), len(yte))

# ---- 다중: Image Segmentation (OpenML segment v1, id 36, 7클래스 각 330)
if not ONLY or "segment_weak" in ONLY:
    sg = fetch_openml(name="segment", version=1, as_frame=True, data_home=args.openml_home, parser="auto")
    X = sg.data.to_numpy(float)
    y = sg.target.astype(str).to_numpy()
    classes = sorted(set(y))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    meta = {"positive_class": None, "negative_class": None, "positive_class_ambiguous": False,
            "detected_classes": classes, "detected_labels": [],
            "class_distribution": {str(k): int(v) for k, v in pd.Series(yte).value_counts().sort_index().items()}}
    model = DecisionTreeClassifier(max_depth=args.segment_depth, random_state=7)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    order = list(model.classes_)
    pred = np.array(order)[proba.argmax(axis=1)]
    df = pd.DataFrame({"sample_id": [f"SG{i:04d}" for i in range(len(yte))], "label": yte, "pred": pred})
    for j, c in enumerate(order):
        df[f"prob_{c}"] = np.round(proba[:, j], 4)
    maps = [("sample_id", "sample_id"), ("label", "y_true"), ("pred", "y_pred")] + [(f"prob_{c}", "prob_per_class") for c in order]
    evaluate("segment_weak", df, "multiclass", maps, ["M1", "M2", "M3", "M4", "M6", "M11", "M12", "M23"], meta)
    print("  split", len(ytr), len(yte), "depth", args.segment_depth)

# ---- 멀티레이블: scene (Mulan 표준 분할 1,211/1,196, 레이블 6개)
SCENE_LABELS = ["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"]


def load(path):
    data, _ = arff.loadarff(path)
    frame = pd.DataFrame(data)
    labels = [c for c in SCENE_LABELS if c in frame.columns]
    assert labels == SCENE_LABELS, frame.columns[-8:]
    X = frame.drop(columns=labels).to_numpy(float)
    Y = frame[labels].apply(lambda s: s.map(lambda v: int(v.decode() if isinstance(v, bytes) else v))).to_numpy()
    return X, Y, labels


if not ONLY or "scene_strong" in ONLY:
    SD = Path(args.scene_dir)
    Xtr, Ytr, labels = load(SD / "scene-train.arff")
    Xte, Yte, _ = load(SD / "scene-test.arff")
    meta = {"positive_class": None, "negative_class": None, "positive_class_ambiguous": False,
            "detected_classes": [], "detected_labels": labels, "class_distribution": {}}
    model = OneVsRestClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)))
    model.fit(Xtr, Ytr)
    proba = model.predict_proba(Xte)
    join = lambda row: "|".join(l for l, v in zip(labels, row) if v)  # noqa: E731
    df = pd.DataFrame({"image_id": [f"SC{i:04d}" for i in range(len(Yte))], "true_labels": [join(r) for r in Yte]})
    for j, l in enumerate(labels):
        df[f"score_{l}"] = np.round(proba[:, j], 4)
    maps = [("image_id", "sample_id"), ("true_labels", "true_labels")] + [(f"score_{l}", "score_per_label") for l in labels]
    evaluate("scene_strong", df, "multilabel", maps, ["M2", "M3", "M4", "M15", "M16", "M17", "M18", "M23"], meta, 0.5)
    print("  split", len(Ytr), len(Yte), "label counts test", Yte.sum(axis=0).tolist())
