import logging
import traceback
from PIL import Image
from io import BytesIO
import base64
import json
import requests

from fastapi import APIRouter, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse
from typing import Optional

from pdf2image import convert_from_bytes


resume_extract_router = APIRouter()
logger = logging.getLogger(__name__)


def encode_image(pil_image: Image.Image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_pdf_to_img_base64(pdf_bytes: bytes) -> list[str]:
    imgs = convert_from_bytes(pdf_bytes)

    base64_imgs = []
    for img in imgs:
        base64_imgs.append(encode_image(img))

    return base64_imgs


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
    cv_file: UploadFile | str,
    prompt_file: Optional[UploadFile] = None,
):
    """
    Receive a PDF or any file via multipart/form-data and return important information in file.

    Args:
        pdf (UploadFile): The uploaded file sent in 'pdf' field.

    Returns:
        JSON response confirming receipt and showing filename.
    """

    if isinstance(cv_file, UploadFile):
        contents = await cv_file.read()
    else:
        contents = requests.get(cv_file)
        contents.raise_for_status()

    if not contents or not cv_file.filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file. Please upload a valid file.",
        )

    filename = cv_file.filename
    logger.info(filename)

    prompt = None
    if prompt_file:
        logger.info("Receive prompt from user")
        prompt = await prompt_file.read()
        prompt = prompt.decode("utf-8")

    try:
        response = await request.app.state.model_gen(
            contents, prompt, "." + filename.split(".")[-1]
        )

        # save_results(response, filename)

        return JSONResponse(
            content={
                "filename": filename,
                "info_extract": response,
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")
