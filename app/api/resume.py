from __future__ import annotations

import traceback
import json
import requests

from loguru import logger
from fastapi import (
    APIRouter,
    UploadFile,
    HTTPException,
    Request,
    status,
    Form,
    File,
    Depends,
)
from fastapi.responses import JSONResponse
from typing import List, Optional, Union
from pydantic import BaseModel
from urllib.parse import urlparse

from app.agent import ResumeService, get_resume_service
from app.core.setting import settings

resume_extract_router = APIRouter()

LEGACY_ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".xlsx", ".xlxs")
STRICT_ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".xlsx")


def save_results(response, filename):
    import os

    path_save = "./project/resume_analysis/data/qwen3-30b-0830"
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    file_path = os.path.join(path_save, filename.split(".")[0] + ".json")
    with open(file_path, "w") as f:
        json.dump(json.loads(response), f, indent=4, ensure_ascii=False)


class ResumeExtractPayload(BaseModel):
    cv_data: Union[str, List[str]]
    cv_embed: list[float]
    file_name: str
    cv_id: str


def _extract_file_name(source_name: str) -> str:
    parsed = urlparse(source_name)
    if parsed.scheme and parsed.path:
        return parsed.path.split("/")[-1] or source_name
    return source_name


def _build_remote_headers() -> dict:
    headers = {}
    if settings.REMOTE_CV_API_KEY:
        headers["x-api-key"] = settings.REMOTE_CV_API_KEY
    return headers


def _validate_magic_bytes(contents: bytes, file_name: str) -> bool:
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        return contents.startswith(b"%PDF")
    if lower_name.endswith(".docx") or lower_name.endswith(".xlsx"):
        return contents.startswith(b"PK\x03\x04")
    if lower_name.endswith(".doc"):
        return contents.startswith(b"\xd0\xcf\x11\xe0")
    if lower_name.endswith(".txt"):
        sample = contents[:2048]
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False


def _validate_resume_payload(
    contents: bytes,
    file_name: str,
    *,
    strict: bool,
):
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    normalized_file_name = _extract_file_name(file_name)
    allowed_extensions = STRICT_ALLOWED_EXTENSIONS if strict else LEGACY_ALLOWED_EXTENSIONS
    if not normalized_file_name.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    if strict and len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large.",
        )

    if strict and not _validate_magic_bytes(contents, normalized_file_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file signature. Please upload a valid file.",
        )


async def _load_resume_request(
    request: Request,
    cv_url: Optional[str],
    cv_file: Optional[UploadFile],
    cv_id: Optional[str],
    *,
    strict: bool = False,
) -> tuple[bytes, str, Optional[str]]:
    content_type = request.headers.get("content-type")
    if content_type and content_type.startswith("application/json"):
        body = await request.json()
        cv_url = body.get("cv_url")
        cv_file = None
        cv_id = body.get("cv_id")
        if not cv_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not provided",
            )

    logger.info(f"Resume ID: {cv_id}")
    if cv_file:
        contents = await cv_file.read()
        file_name = cv_file.filename or ""
    elif cv_url:
        response = requests.get(
            cv_url,
            headers=_build_remote_headers(),
            timeout=settings.REMOTE_CV_TIMEOUT,
        )
        response.raise_for_status()
        contents = response.content
        file_name = _extract_file_name(cv_url)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not provided",
        )

    _validate_resume_payload(contents, file_name, strict=strict)
    logger.info(file_name)
    return contents, file_name, cv_id


@resume_extract_router.post("/check-duplication")
async def check_duplication(
    request: Request,
    cv_url: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    cv_id: Optional[str] = Form(None),
    resume_service: ResumeService = Depends(get_resume_service),
):
    try:
        contents, file_name, _ = await _load_resume_request(
            request, cv_url, cv_file, cv_id, strict=False
        )
        check_result, cv_data_converted, emb_result = (
            await resume_service.check_duplication(contents, file_name)
        )

        return JSONResponse(
            content={
                "file_name": file_name,
                "check_result": check_result,
                "cv_data_converted": cv_data_converted,
                "emb_result": emb_result,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@resume_extract_router.post("/extract")
async def extract(
    request: Request,
    cv_url: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = None,
    cv_id: Optional[str] = Form(None),
    resume_service: ResumeService = Depends(get_resume_service),
):
    sys_mess = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        sys_mess = prompt.decode("utf-8")

    try:
        contents, file_name, _ = await _load_resume_request(
            request, cv_url, cv_file, cv_id, strict=False
        )
        gen_res, gen_res_format, service_resp = await resume_service.extract(
            contents, sys_mess, file_name
        )

        return JSONResponse(
            content={
                "file_name": file_name,
                "info_extract": gen_res_format,
                "info_extract_raw": gen_res,
                "service_resp": service_resp.model_dump(),
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@resume_extract_router.post("/ingest")
async def ingest(
    request: Request,
    cv_url: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = None,
    cv_id: Optional[str] = Form(None),
    resume_service: ResumeService = Depends(get_resume_service),
):
    sys_mess = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        sys_mess = prompt.decode("utf-8")

    try:
        contents, file_name, cv_id = await _load_resume_request(
            request, cv_url, cv_file, cv_id, strict=True
        )
        ingest_result = await resume_service.ingest_resume(
            contents, file_name, cv_id=cv_id, sys_mess=sys_mess
        )
        return JSONResponse(content=ingest_result)

    except HTTPException:
        raise
    except requests.RequestException as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@resume_extract_router.post("/extract-store")
async def extract_and_store(
    payload: ResumeExtractPayload,
    resume_service: ResumeService = Depends(get_resume_service),
):
    try:
        gen_res, gen_res_format = await resume_service.extract_and_store(
            payload.cv_data, payload.file_name, payload.cv_id, payload.cv_embed
        )

        return JSONResponse(
            content={
                "file_name": payload.file_name,
                "info_extract": gen_res_format,
                "info_extract_raw": gen_res,
                "cv_id": payload.cv_id,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@resume_extract_router.put("/delete")
async def delete_resume(
    request: Request,
    resume_service: ResumeService = Depends(get_resume_service),
):
    try:
        content_type = request.headers.get("content-type")
        if content_type and content_type.startswith("application/json"):
            body = await request.json()
            cv_id = body.get("cv_id")

            return await resume_service.del_cv(cv_id=cv_id)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
