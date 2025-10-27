import logging
import requests
import traceback

from fastapi import APIRouter, UploadFile, HTTPException, Request, status, Form, File
from fastapi.responses import JSONResponse
from typing import Optional

from ..agent import JDService


jd_matcher_router = APIRouter()
logger = logging.getLogger(__name__)


@jd_matcher_router.post("/upload")
async def extract(
    request: Request,
    jd_file: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = None,
    jd_id: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type")
    if content_type and content_type.startswith("application/json"):
        body = await request.json()
        contents = body.get("jd_content")
        jd_id = body.get("jd_id")

    logger.info(f"Job description ID: {jd_id}")
    if jd_file:
        contents = await jd_file.read()
        file_name = jd_file.filename

    # elif jd_url:
    #     contents = requests.get(jd_url, timeout=30)
    #     contents.raise_for_status()
    #     contents = contents.content
    #     file_name = jd_url

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not provided",
        )

    if not contents: # or not file_name.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    logger.info(file_name)

    prompt = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        prompt = prompt.decode("utf-8")

    try:
        jd_service = JDService()
        gen_res, top_cv_id = await jd_service.extract_and_match(
            contents, prompt, file_name, jd_id
        )

        return JSONResponse(
            content={
                "filename": file_name,
                "top_cv_id": top_cv_id,
                "info_extract_raw": gen_res,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        # return
