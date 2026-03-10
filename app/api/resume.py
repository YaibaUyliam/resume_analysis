# import logging
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
from typing import Optional
from pydantic import BaseModel

from app.agent import ResumeService, get_resume_service

resume_extract_router = APIRouter()
# logger = logging.getLogger(__name__)


# def encode_image(pil_image: Image.Image):
#     buffer = BytesIO()
#     pil_image.save(buffer, format="JPEG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_results(response, filename):
    import os

    path_save = "./project/resume_analysis/data/qwen3-30b-0830"
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    file_path = os.path.join(path_save, filename.split(".")[0] + ".json")
    with open(file_path, "w") as f:
        json.dump(json.loads(response), f, indent=4, ensure_ascii=False)


class ResumeExtractPayload(BaseModel):
    cv_data: str | list[str]
    cv_embed: list[float]
    file_name: str
    cv_id: str


@resume_extract_router.post("/check-duplication")
async def check_duplication(
    request: Request,
    cv_url: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    cv_id: Optional[str] = Form(None),
    resume_service: ResumeService = Depends(get_resume_service),
):
    content_type = request.headers.get("content-type")
    if content_type and content_type.startswith("application/json"):
        body = await request.json()
        cv_url = body.get("cv_url")
        cv_file = None  # None
        cv_id = body.get("cv_id")

        if not cv_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not provided",
            )

    logger.info(f"Resume ID: {cv_id}")
    if cv_file:
        contents = await cv_file.read()
        file_name = cv_file.filename

    elif cv_url:
        # contents = requests.get(cv_url, timeout=30)
        headers = {"x-api-key": "CgLju4wmGd9ervXbhk2nJMDSsUzcRpPy"}
        contents = requests.get(cv_url, headers=headers, timeout=30)
        contents.raise_for_status()
        contents = contents.content
        file_name = cv_url

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not provided",
        )

    if not contents or not file_name.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    logger.info(file_name)

    try:
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
    content_type = request.headers.get("content-type")
    if content_type and content_type.startswith("application/json"):
        body = await request.json()
        cv_url = body.get("cv_url")
        cv_file = None  # None
        cv_id = body.get("cv_id")

        if not cv_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not provided",
            )

    logger.info(f"Resume ID: {cv_id}")
    if cv_file:
        contents = await cv_file.read()
        file_name = cv_file.filename

    elif cv_url:
        # contents = requests.get(cv_url, timeout=30)
        headers = {"x-api-key": "CgLju4wmGd9ervXbhk2nJMDSsUzcRpPy"}
        contents = requests.get(cv_url, headers=headers, timeout=30)
        contents.raise_for_status()
        contents = contents.content
        file_name = cv_url

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not provided",
        )

    if not contents or not file_name.endswith(
        (".pdf", ".docx", ".txt", ".doc", ".xlxs")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    logger.info(file_name)

    sys_mess = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        sys_mess = prompt.decode("utf-8")

    try:
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


@resume_extract_router.post("/extract-store")
async def extract_and_store(
    payload: ResumeExtractPayload,
    resume_service: ResumeService = Depends(get_resume_service),
):
    try:
        gen_res, gen_res_format, _ = await resume_service.extract_and_store(
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
