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
        file_name = None

    logger.info(f"Job description ID: {jd_id}")
    if jd_file:
        contents = await jd_file.read()
        file_name = jd_file.filename

    if not contents:  # or not file_name.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    prompt = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        prompt = prompt.decode("utf-8")

    try:
        jd_service = JDService()
        gen_res, top_cv = await jd_service.extract_match_review(
            contents, prompt, file_name, jd_id
        )

        top_cv_shorten = []
        for v in top_cv:
            try:
                source = v["_source"]
                filtered_dict = {}

                filtered_dict["cv_id"] = source["id"]
                filtered_dict["cv_url"] = source["cv_url"]
                filtered_dict["content"] = source["content"]
                filtered_dict["year_of_experience"] = source.get(
                    "year_of_experience"
                )
                filtered_dict["full_name"] = source["full_name"]
                filtered_dict["match_score"] = v["match_score"]
                filtered_dict["strong_matches"] = v["strong_matches"]
                filtered_dict["partial_matches"] = v["partial_matches"]
                filtered_dict["missing_keywords"] = v["missing_keywords"]
                filtered_dict["review"] = v["summary"]

                top_cv_shorten.append(filtered_dict)

            except:
                logger.error(traceback.format_exc())

        return JSONResponse(
            content={
                "top_cv": top_cv_shorten,
                "info_extract_raw": gen_res,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        # return
