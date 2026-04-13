"""
main.py — FastAPI 앱의 진입점
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

from schemas import (
    AnalysisResponse, TaskType,
    ConfirmMappingRequest, ConfirmMappingResponse,
)
from analyzer import parse_file_content, analyze_columns_with_llm
from validator import validate_mapping


load_dotenv()

ALLOWED_EXTENSIONS = {"csv", "json"}

openai_client: AsyncOpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "backend/.env 파일에 OPENAI_API_KEY=sk-... 를 추가해주세요."
        )
    openai_client = AsyncOpenAI(api_key=api_key)
    print("✅ OpenAI 클라이언트 초기화 완료")
    yield
    print("🛑 서버 종료")


app = FastAPI(
    title="ISO 4213 AI 분류 모델 평가 API",
    description=(
        "CSV 또는 JSON 파일을 업로드하면 "
        "LLM(GPT-4.1-nano)이 ISO/IEC TS 4213:2022 기준으로 컬럼 역할을 자동 매핑합니다."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok"}


@app.post(
    "/api/analyze-columns",
    response_model=AnalysisResponse,
    summary="컬럼 자동 매핑",
    description=(
        "**task_type** (binary / multiclass / multilabel) 을 사용자가 직접 지정하면, "
        "LLM이 해당 유형의 ISO 4213 컬럼 정의에 맞게 각 컬럼을 매핑합니다.\n\n"
        "지원 파일 형식: **CSV**, **JSON**"
    ),
    tags=["Column Analysis"],
)
async def analyze_columns(
    task_type: TaskType = Form(
        ...,
        description="분류 모델 유형: binary / multiclass / multilabel",
    ),
    file: UploadFile = File(
        ...,
        description="분석할 파일 (.csv 또는 .json)",
    ),
) -> AnalysisResponse:
    """
    [POST] /api/analyze-columns

    Request: multipart/form-data
      - task_type: "binary" | "multiclass" | "multilabel"
      - file: CSV 또는 JSON 파일

    Response: AnalysisResponse JSON
    """
    # 파일 확장자 검증
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: .{ext or '(없음)'}. CSV 또는 JSON 파일을 업로드해주세요."
        )

    # 파일 내용 읽기
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    # 파일 파싱 (CSV / JSON 공통 처리)
    try:
        columns, df = parse_file_content(file_content, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"파일 파싱 실패: {str(e)}")

    if not columns:
        raise HTTPException(status_code=422, detail="파일에 컬럼이 없습니다.")

    # LLM 호출 → 컬럼 매핑 + 메타데이터 추출
    try:
        result = await analyze_columns_with_llm(
            client=openai_client,
            task_type=task_type,
            columns=columns,
            df=df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 분석 실패: {str(e)}")

    return result


@app.post(
    "/api/confirm-mapping",
    response_model=ConfirmMappingResponse,
    summary="매핑 확정 및 유효성 검사",
    description=(
        "사용자가 검토·수정한 컬럼 매핑을 서버에 제출합니다.\n\n"
        "- 필수 역할 누락, 중복 매핑 등 **오류(errors)** 를 반환합니다.\n"
        "- 선택 역할 미설정으로 인한 TC 범위 축소 **경고(warnings)** 를 반환합니다.\n"
        "- 현재 매핑으로 **계산 가능한 TC 목록(available_tcs)** 을 반환합니다.\n"
        "- `is_valid=true` 이면 TC 계산 단계로 진행 가능합니다."
    ),
    tags=["Column Analysis"],
)
async def confirm_mapping(
    request: ConfirmMappingRequest,
) -> ConfirmMappingResponse:
    """
    [POST] /api/confirm-mapping

    Request: JSON body
      - task_type: "binary" | "multiclass" | "multilabel"
      - column_mappings: [{column, role}, ...]

    Response: ConfirmMappingResponse JSON
      - is_valid: bool
      - errors: 치명적 오류 목록
      - warnings: 경고 목록
      - available_tcs: 계산 가능한 TC
      - unavailable_tcs: 계산 불가 TC (이유 포함)
      - confirmed_mappings: 확정된 매핑 (그대로 반환)
    """
    return validate_mapping(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
