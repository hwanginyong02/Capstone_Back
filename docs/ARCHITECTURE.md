# 백엔드 구조 한눈에 보기 (처음 오신 분 먼저 읽기)

이 문서 하나면 **5분 안에** 이 백엔드가 뭘 하고, 코드가 어디 있고, 새 작업을 어디서 시작할지 알 수 있습니다. 코드를 처음 연다는 가정으로 씁니다.

---

## 1. 이 백엔드가 하는 일 (한 문단)

**ISO/IEC TS 4213** 기준으로 AI **분류 모델**의 성능을 평가하고 성적서를 발급하는 FastAPI 서버입니다. 사용자가 모델 예측 결과 파일(CSV/JSON)을 올리면 → ① **어떤 컬럼이 정답/예측/점수인지 파악**하고(LLM 자동 매핑) → ② **지표(정확도·정밀도 등)를 계산**하고 → ③ **결과를 자연어로 설명**하고 → ④ **성적서를 발급**합니다. Render + Neon Postgres에 배포됩니다.

---

## 2. 딱 한 장으로 보는 구조

> **핵심 규칙 하나만 기억하세요: `app/` 아래 폴더 = "하는 일(도메인)" 하나.**
> 그리고 **어느 도메인이든 파일 구성이 똑같습니다** → `router`(API) → `service`(흐름 조율) → 나머지(세부 로직) + `schemas`(그 도메인의 데이터 형태).

```
app/
├── main.py            앱 진입점 (uvicorn app.main:app) — 라우터 5개 연결, 시작 시 DB 준비
│
├── core/              🔧 모든 도메인이 공유하는 것
│   ├── schemas.py       공용 데이터 규격(enum·역할규칙·지표요건·"분석→평가" 인계 모델)
│   ├── database.py      DB 연결(SQLAlchemy) — 로컬 SQLite / 배포 Postgres
│   └── parsing.py       업로드 파일(CSV/JSON) → 표(DataFrame) 변환 (공용 유틸)
│
├── analysis/          ① 분석 — 파일의 어떤 컬럼이 뭔지 파악 + 데이터 점검
│   ├── router.py            API: 컬럼 매핑 (/api/analyze-columns, /api/confirm-mapping)
│   ├── validation_router.py API: 데이터 검증 (/api/validate-data)
│   ├── analysis_service.py    흐름 조율(무키면 규칙매핑, LLM 실패하면 규칙매핑으로 강등)
│   ├── validation_service.py  데이터 검증 흐름 조율
│   ├── llm_mapper.py     LLM으로 컬럼 자동 매핑        fallback_mapper.py  규칙 기반 매핑
│   ├── reconcile.py      LLM이 준 컬럼명을 실제 헤더에 맞춤   metadata.py  클래스·분포 추출
│   ├── prompt_builder.py 매핑용 LLM 프롬프트          validator.py  매핑 유효성·계산가능 지표
│   ├── validation_checks.py  데이터 점검 항목들       schemas.py  이 도메인 데이터 형태
│
├── evaluation/        ② 평가 — 지표 계산
│   ├── router.py        API (/api/evaluate)          service.py  평가 흐름 조율
│   ├── engine.py        지표 → 계산함수로 분배          preprocessor.py  계산 전 데이터 정리
│   ├── report.py        결과 성공/실패 정리           schemas.py
│   └── metrics/         실제 지표 계산(sklearn): common·binary·multiclass·multilabel
│
├── narrative/         ③ 서술 — 결과를 사람이 읽을 글로 (LLM)
│   ├── router.py        API (/api/generate-narrative)  service.py  서술 생성 흐름 조율
│   ├── grounding.py     환각 방어(없는 숫자 못 쓰게)   derived.py  서버 파생 계산
│   ├── prompt.py        서술 프롬프트                 fallback.py  LLM 실패 시 규칙 서술
│   └── baselines.py     지표 참조 기준치              schemas.py
│
└── issuance/          ④ 발급 — 성적서 번호 채번·발급 (유일하게 DB 사용)
    ├── router.py        API (/api/organization, /api/reports/*)  service.py  발급 트랜잭션
    ├── models.py        DB 테이블(Organization·Report·Issuance)  serializers.py  DB→응답 변환
    └── bootstrap.py     기본 기관 시드                schemas.py
```

> 💡 각 폴더의 `__init__.py`를 열면 그 도메인의 **파일 목차**가 주석으로 정리돼 있고, 각 `.py` 파일 맨 위에는 **"이 파일이 뭐고 뭐랑 상호작용하는지"** 헤더 주석이 있습니다.

---

## 3. 요청이 흐르는 순서

사용자가 파일을 올리면 도메인을 **순서대로** 거칩니다. 이 흐름 = 위 폴더 순서입니다.

```
[예측결과 파일 업로드]
      │
      ▼
① /api/analyze-columns   analysis : 컬럼 역할 자동 파악 (LLM, 없으면 규칙)
      ▼
① /api/confirm-mapping   analysis : 사용자가 고른 매핑 검사 → 계산 가능한 지표 목록
      ▼  (선택) /api/validate-data  analysis : 계산 전 데이터 점검(결측·중복 등)
      ▼
② /api/evaluate          evaluation : 지표 계산
      ▼
③ /api/generate-narrative narrative : 계산 결과를 자연어로 서술
      ▼
④ /api/reports/issue     issuance : 성적서 번호 발급 + DB 저장
```

- **LLM(OpenAI)을 쓰는 곳**: ①분석·③서술. 키(`OPENAI_API_KEY`)가 없으면 자동으로 규칙 기반으로 대체돼서 서버는 안 죽습니다.
- **DB를 쓰는 곳**: ④발급 뿐. 나머지는 업로드 파일을 메모리에서 처리하고 끝(디스크 저장 없음).

---

## 4. 도메인 요약표

| 도메인 | 하는 일 | 주요 API | "진입점" 파일 |
|---|---|---|---|
| **core** | 공용 규격·DB·파일파싱 | — | `schemas.py`, `database.py` |
| **analysis** | 컬럼 파악 + 데이터 점검 | `/api/analyze-columns`, `/api/confirm-mapping`, `/api/validate-data` | `router.py` → `analysis_service.py` |
| **evaluation** | 지표 계산 | `/api/evaluate` | `router.py` → `service.py` |
| **narrative** | 결과 서술 | `/api/generate-narrative` | `router.py` → `service.py` |
| **issuance** | 성적서 발급 | `/api/organization`, `/api/reports/*` | `router.py` → `service.py` |

시스템: `GET /health` (상태 확인).

---

## 5. 새 작업을 어디서 시작하나 (신규 개발자용)

| 하고 싶은 것 | 열 파일 |
|---|---|
| API 동작/입출력 바꾸기 | 해당 도메인 `router.py` (얇음 — 검증·상태코드만) |
| 비즈니스 로직(흐름) 바꾸기 | 해당 도메인 `service.py` |
| 요청/응답 필드 추가 | 해당 도메인 `schemas.py` (공용이면 `core/schemas.py`) |
| 새 지표 추가 | `evaluation/metrics/*` + `evaluation/engine.py`의 레지스트리 |
| 서술 문구/규칙 | `narrative/prompt.py`(LLM) 또는 `narrative/fallback.py`(규칙) |
| DB 테이블 | `issuance/models.py` |

**패턴이 항상 같습니다**: 요청은 `router`(HTTP만) → `service`(흐름) → 세부 헬퍼로 흐릅니다. 그래서 어느 도메인이든 같은 방식으로 읽힙니다.

---

## 6. 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload      # → http://127.0.0.1:8000/docs (Swagger로 모든 API 확인)
pytest -q                          # 테스트 (tests/)
```

환경변수: `OPENAI_API_KEY`(없으면 규칙 기반 동작), `DATABASE_URL`(없으면 로컬 SQLite `data/app.db`).

---

## 7. 알아두면 좋은 규칙

1. **import는 절대 경로** (`from app.analysis.llm_mapper import ...`).
2. **의존 방향은 항상 도메인 → core** (core는 도메인을 모름 → 순환 없음). 예외: 평가가 분석의 파싱/검증을 재사용(상류 참조)하는 것은 허용.
3. **`app/__init__.py`·`app/core/__init__.py`에 `import`를 넣지 마세요** — `main.py`의 `load_dotenv()`가 DB import보다 먼저 실행돼야 하는 순서가 깨집니다(파일 주석 참조).
4. **DB 스키마는 `init_db()`의 `create_all`로 생성**(마이그레이션 도구 없음).
5. 배포 구성은 [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) 참조.

---

## 8. 알려진 한계 (2026-09-07 확정)

여기 적힌 것들은 **버그가 아니라 의식적으로 그은 선**이다. 각 항목에 왜 그렇게 정했는지를
남긴다 — 근거를 잃으면 다음 사람이 같은 논의를 처음부터 다시 한다.
대응 항목 번호는 `ISSUES.md` 를 가리킨다.

### 8.1 성적서의 진본성 — 서버가 완전히 보증하지는 않는다 (F-02·F-03·F-05)

발급 시점의 성적서 원본(JSON + SHA-256)과 기관 스냅샷은 서버에 보관된다. 번호로
복원할 수도 있다. 그러나 **다음 셋은 구현하지 않았다.**

| # | 한계 | 왜 이번 범위에서 제외했나 |
|---|---|---|
| **F-02** | 평가와 발급이 서버에서 결속되지 않는다 — 평가를 거치지 않고도 정식 번호를 받을 수 있다 | 백엔드 단독으로 닫히지 않는다(프론트 필수·3단 롤아웃). 무엇보다 **기존 사용자 전원이 발급 불가**가 되고, 결속을 넣어도 **2행짜리 CSV 로 평가를 통과시켜 우회**할 수 있어 실효가 낮다 |
| **F-03** | `/api/generate-narrative` 가 클라이언트가 보낸 `fact_sheet` 를 무검증 신뢰한다 | F-02 와 같은 뿌리다. 서버가 평가 결과를 갖고 있어야 대조할 수 있는데 그것이 F-02 다 — 따로 닫으면 반쪽이 된다 |
| **F-05** | 서버가 PDF 를 만들지도 보관하지도 않는다. 최종 산출물은 브라우저 인쇄물이다 | 서버 렌더링 스택(헤드리스 브라우저)이 필요하고 Render free 512 MB 안에서 운영 위험이 크다 |

**실질적 의미**: 발급된 성적서의 *내용*은 서버 사본으로 검증할 수 있지만, *그 성적서가
실제 평가를 거쳤는가*는 서버가 보증하지 않는다. 대외 제출용으로 쓸 때 이 점을 알고 써야 한다.

#### 6차 라운드 재조사 — 위 판단의 근거는 전부 사실이고, 두 곳은 **더 강했다**

착수 여부를 다시 물었고 "착수하지 않는다"로 확정했다(2026-09-07). 그 결정의 근거를 실측으로
다시 확인한 결과다 — 다음 사람이 같은 조사를 반복하지 않도록 남긴다.

- **우회 비용은 '2행 CSV' 보다 낮다.** 행 수 하한은 `preprocessor.py` 의 `df.empty` 와
  결측 제거 후 `len(df) == 0` 두 개뿐이다. 즉 **헤더 + 데이터 1행**이면 평가가 성공한다.
- **기존 run 은 플래그로 구제할 수 없다.** `rawFile` 은 설계상 persist 대상이 아니고
  (`useWorkflowStore` 의 `partialize`), 재수화 시 `needsFileReupload` 가 선다. 발급에 평가
  토큰을 요구하는 어떤 설계도 **변경 이전에 만들어진 run 전부를 영구히 발급 불가로 만든다** —
  grandfather 플래그로도 우회할 수 없다(사용자가 동일 파일을 다시 올려야 한다).
- **저장된 입력으로 재계산하는 우회로는 없다.** 스냅샷이 보관하는 것은 *성적서*
  (`FinalReportData` JSON)이고 데이터셋이 아니다 — `datasetSamples` 는 항상 `[]` 이며
  (`mapWorkflowToFinalReport`), 백엔드는 업로드 파일을 디스크에 쓰지 않는다.
- **F-03 에는 값싼 부분 승리가 있다 — 이번에는 하지 않는다.** `fact_sheet` 의 **산술 정합성**은
  F-02 결속 없이도 서버가 혼자 검사할 수 있다(accuracy = trace/total, score = 통과/대상,
  각 `status` 가 값-목표 비교와 일치, verdict 가 status 들에서 유도). 다만 착수하려면 아래
  넷을 반드시 지켜야 한다 — 그러지 않으면 정상 사용자의 성적서가 깨진다.
  · 검사 위치는 `NarrativeRequest`(또는 서비스)다. `FactSheet` 에는 `task_type` 이 없어
    multilabel 조건부 규칙이 판정 불가가 된다.
  · 판정 비교는 **비대칭**이어야 한다. `computeVerdict` 는 핵심지표가 `unavailable` 이면
    PASS 를 CONDITIONAL_PASS 로 내리는데 `buildFactSheet` 는 `unavailable` 지표를 페이로드에서
    **빼기 때문에**, 서버는 그 사유를 볼 수 없어 PASS 를 계산한다. 동등 비교는 오탐이 된다.
  · `n_samples` == 혼동행렬 합은 **항등식이 아니다.** multilabel 은 전 레이블 합이라
    `n_samples × 레이블 수` 다(B-02 와 같은 뿌리).
  · multiclass 의 macro P/R/F1 은 제출된 행렬에서 **정확히 재계산되지 않는다** —
    행렬 라벨은 `y_true ∪ y_pred`(유령 클래스 포함)인데 macro 평균은 `y_true` 기준이다.
- **F-03 만 닫아도 발급본이 신뢰 가능해지지는 않는다.** `POST /api/reports/issue` 는
  `content: dict` 를 크기 상한(1 MiB)만 보고 받는다 — 의미 검증이 없어 서술 파이프라인
  전체를 건너뛸 수 있다. 진본성의 무게는 결국 F-02 에 있다.

### 8.2 운영·인프라

| # | 한계 | 판단 |
|---|---|---|
| **G-03** | LLM 예산 상한이 **프로세스 로컬**이라 워커가 N개면 실효 상한도 N배 | 과허용 방향으로만 틀리므로 정상 사용자를 잘못 막지 않는다. 정확한 총량이 필요해지면 DB 카운터로 옮긴다 |
| **G-03** | 인증·레이트리밋 없음 | 인증은 넣지 않기로 확정(★결정 9). IP 리밋은 **프록시 뒤 실제 client IP 가 미관측**이라 보류 — 모든 요청이 같은 IP 로 보이면 IP 리밋이 전역 리밋으로 붕괴해 정상 사용자 1명이 전체를 막는다 |
| **F-11** | 마이그레이션 도구가 없다(`create_all` 만) | **기존 테이블에 컬럼을 추가하지 않는다**를 규칙으로 둔다. 신규 테이블은 `create_all` 이 정상 생성하므로(실증) 스키마 확장은 새 테이블로 한다 |
| **F-12** | 동시 채번 직렬화(`BEGIN IMMEDIATE`)가 SQLite 전용이라 PostgreSQL 에는 적용되지 않는다 | 재시도 5회 + 유니크 제약이 실질 방어선이다. 동시 발급이 드물어 현행 유지 |
| **H-06** | PostgreSQL 경로가 테스트·CI 어디서도 실행되지 않는다 | 로컬에 `psycopg2` 가 없고 CI 에 DB 서비스를 붙이지 않았다. **로컬 초록은 배포 조합의 증거가 아니다** |
| **H-05** | 로컬 pytest 가 프로젝트 venv 가 아니라 전역 anaconda 에서 돈다(fastapi/pydantic 버전이 `requirements.txt` 와 다름) | 위와 같은 이유로 **배포 조합의 증거는 CI 뿐**이다 |
| **G-07** | `DATABASE_URL` 이 없으면 휘발성 SQLite 로 내려가고 기동을 계속한다 | 하드 실패 가드는 `REQUIRE_PERSISTENT_DB=1` 로 **기본 꺼짐**이다. 프로덕션이 이미 SQLite 로 돌고 있다면 켜는 순간 서비스가 죽기 때문. 대신 `/health` 가 평시에도 `persistent` 를 싣고 기동 시 경고를 남긴다 |

### 8.3 서술(LLM) 품질

| # | 한계 | 판단 |
|---|---|---|
| **G-05** | grounding 이 **숫자 토큰**을 대조한다. 판정과 직접 모순되는 주장은 별도로 잡지만, 완곡한 표현이나 영어 서술은 잡지 못한다 | 일반적인 '금지 표현 사전'은 문체 취향이라 도입하지 않는다. 사실 관계(판정 모순)만 막는다 |

### 8.4 프론트 측 한계 (`Capstone_Front`)

| # | 한계 | 판단 |
|---|---|---|
| **E-10** | 워크스페이스 저장소에 크로스탭 동기화가 없어 두 탭을 함께 쓰면 나중 쓰기가 상대 탭의 run 을 덮는다 | 단일 탭 사용이 정상 흐름이다. `storage` 이벤트 동기화는 다음 범위 |
| **E-11** | run 하나당 수십만 바이트가 localStorage 에 쌓여 quota 초과 시 저장이 실패할 수 있다 | 저장 실패가 진행 중인 입력을 잃게 하지는 않는다(try/catch). 서버 보관본이 생겼으므로(F-01) 위험은 종전보다 낮다 |
| **E-15** | 평가·검증 요청에 타임아웃·재시도가 없다(컬럼 분석에만 있다) | 평가·검증은 사용자가 결과를 기다리는 전면 작업이라 조용한 재시도가 오히려 혼란스럽다. 현행 유지 |
| **E-14** | LLM 서술 생성 중(최대 160초) 이탈했다가 재진입하면 평가와 서술이 다시 실행된다 | 캐시 히트 조건이 `isEvaluated`(서술 병합 완료)라 그렇다. 미완 상태를 캐시하면 반쪽 성적서가 완성본으로 굳는다 — 재실행이 안전한 쪽이다 |
| **H-03** | `useReportData` 는 성적서 입력을 `run.workflowSnapshot` 이 아니라 **살아 있는 워크플로우 store** 에서 읽는다 | 6차 라운드에 **소유권 가드**를 넣어 실해를 닫았다(아래 8.5). 입력원 자체를 스냅샷으로 바꾸려면 `decisionThreshold` 를 스냅샷 타입에 추가해야 하고, 그 타입은 localStorage 에 영속된 `WorkspaceEvaluationRun.workflowSnapshot` 이라 마이그레이션이 따라온다. 가드로 닫히는 실해에 비해 비용이 크다 |

### 8.5 규칙 사본과 교차 저장소 드리프트 (A-03 · 6차 라운드 갱신)

지표→필수컬럼 규칙은 네 곳에 있다 — SPEC.md · 백엔드 `METRIC_REQUIREMENTS` ·
프론트 `REQUIRED_COLUMNS_BY_METRIC` · 프론트 `METRICS`. **사본을 하나로 줄이지 않기로 한다.**

**왜 API 로 내려보내지 않는가.** 그 방향은 약 26개 파일과 두 저장소의 **순서 있는 배포**가
필요하고, 워크플로우 화면 셋과 **랜딩 페이지**를 동기 렌더에서 fetch 의존으로 바꾼다
(백엔드는 Render **free** 인스턴스이고, 이 저장소의 CI 자체가 배포 후 `/health` 응답에
최대 10분을 예산으로 잡는다). 게다가 소비처 둘은 애초에 await 할 수 없다 —
`reportShowcase.ts` 는 **모듈 로드 시점**에 값을 계산하고, zustand persist 의 `migrate` 는
동기 함수다. 그러면서도 없앨 수 있는 사본은 최대 둘뿐이다: SPEC.md 는 산문이고
`METRICS` 는 성적서가 인쇄하는 이름·산식을 갖고 있다.

**대신 6차가 닫은 것 — 검증된 구멍 하나.** 5차는 "계약 테스트가 드리프트를 CI 에서 즉시
red 로 잡는다"고 기록했다. **대체로는 맞는데 한 방향이 비어 있었다.** 실측으로 구분한 결과다
(각 항목을 실제로 변이해 전체 스위트를 돌렸다).

프론트 계약 테스트는 프론트 표를 **같은 파일 안의 손 사본**(`BACKEND_METRIC_REQUIREMENTS`)과
비교하고, 두 저장소의 CI 는 서로를 checkout 하지 않는다. 그래서 **백엔드 단독 변경에 대한
방어선은 전부 백엔드 쪽 테스트다.** 그 방어선의 실측 범위 —

| 백엔드 표를 이렇게 바꾸면 | 기존 스위트 | 무엇이 잡나 |
|---|---|---|
| 지표를 **삭제** (binary M20 제거) | **red** (1 failed) | `evaluate_*` 골든 — 골든 요청이 그 task 의 전 지표를 선택하므로 값이 사라진다 |
| 요구 역할 **변경** (M1 에 sample_id 추가) | **red** (7 failed) | `test_metric_requirements.py` 의 충분·필요 대조 + 확률 지원 테스트 |
| `VALID_ROLES_BY_TASK` 에서 확률 역할 제거 | **red** (20 failed) | 골든 + 폴백 역할 테스트 |
| `PREDICTION_ROLES_BY_TASK` 의 대체 역할 제거 | **red** (2 failed) | `test_probability_support.py` |
| 지표를 **추가** (binary 에 M14 추가) | **초록 (476 passed)** | **없었다** ← 이것이 구멍이다 |

추가가 빠져나가는 이유는 셋이 겹친다: 충분·필요 대조는 **표를 순회**하므로 새 항목도 구현이
있으면 통과하고, 골든의 `selected_metric_ids` 는 **별도 리터럴**이라 새 지표를 고르지 않으며,
프론트의 "노출 지표 ⊆ 백엔드 표" 단정은 **부분집합** 방향이라 백엔드가 더 많이 갖는 것을
문제로 보지 않는다.

직접 피해는 크지 않다(UI 가 계산 가능한 지표를 제공하지 않을 뿐이다). 문제는 **아무 신호 없이
두 표가 벌어진 상태로 남는다**는 것이다 — 그 다음 변경(그 지표의 역할 변경·삭제)이 낡은 프론트
표와 만나면 5단계 '다음' 게이트가 프론트 **로컬 표로** 판정하므로 백엔드가 받아줄 매핑을 화면이
하드 차단한다. A-13 이 정확히 그 실패 형태였다. SPEC.md 도 같이 낡는다.

`tests/test_metric_requirements_pin.py` 가 표를 **명시적 기대값에 고정**한다. 값 비교가
목적이 아니라 **백엔드 표를 건드리면 반드시 그 파일도 건드리게 만드는 것**이 목적이고,
실패 메시지가 함께 고쳐야 할 프론트 파일 두 개를 이름으로 알려준다. 위 5가지 변이 전부를
이 고정핀이 red 로 만드는 것을 확인했다.

**남은 사본 관계 하나 더 — 서술 기준치의 키가 프론트 지표 이름이다.**
`app/narrative/baselines.py` 의 `BASELINES`·`LOWER_IS_BETTER` 는 프론트
`METRICS[].name`(display name)으로 키를 잡는다. 프론트에서 지표 이름을 바꾸면 백엔드가
기준치를 못 찾아 8절 벤치마크 문장에서 그 지표가 조용히 빠지고, `LOWER_IS_BETTER` 에서
빠지면 **낮을수록 좋은 지표가 반대로 판정된다.** 백엔드는 프론트 파일을 읽을 수 없으므로
프론트 쪽에 고정핀을 둔다(`src/data/narrativeBaselineNames.test.ts`).
