import logging
import traceback
from PIL import Image
from io import BytesIO
import base64
import json
import requests

from fastapi import APIRouter, UploadFile, HTTPException, Request, status, Form, File
from fastapi.responses import JSONResponse
from typing import Optional

from .utils import convert_resume_format


resume_extract_router = APIRouter()
logger = logging.getLogger(__name__)


def encode_image(pil_image: Image.Image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_results(response, filename):
    import os

    path_save = "/home/yaiba/project/resume_analysis/data/qwen3-30b-0830"
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    file_path = os.path.join(path_save, filename.split(".")[0] + ".json")
    with open(file_path, "w") as f:
        json.dump(json.loads(response), f, indent=4, ensure_ascii=False)


@resume_extract_router.post("/extract")
async def extract(
    request: Request,
    cv_url: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = None,
):
    """
    Receive a PDF or any file via multipart/form-data and return important information in file.

    Args:
        pdf (UploadFile): The uploaded file sent in 'pdf' field.

    Returns:
        JSON response confirming receipt and showing filename.
    """

    if cv_file:
        contents = await cv_file.read()
        file_name = cv_file.filename

    elif cv_url:
        contents = requests.get(cv_url, timeout=30)
        contents.raise_for_status()
        contents = contents.content
        file_name = cv_url

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not provided",
        )

    if not contents or not file_name.endswith((".pdf", ".docx", ".doc")):
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
        response = await request.app.state.model_gen(
            contents, prompt, "." + file_name.split(".")[-1]
        )

        # save_results(response, filename)

        return JSONResponse(
            content={
                "filename": file_name,
                "info_extract": convert_resume_format(response),
                "info_extract_raw": response,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")
