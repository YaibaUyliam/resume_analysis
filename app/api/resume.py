import logging
import traceback
from PIL import Image
from io import BytesIO
import base64
import json

from fastapi import APIRouter, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

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


@resume_extract_router.post("/extract")
async def extract(request: Request, cv_file: UploadFile):
    """
    Receive a PDF or any file via multipart/form-data and return important information in file.

    Args:
        pdf (UploadFile): The uploaded file sent in 'pdf' field.

    Returns:
        JSON response confirming receipt and showing filename.
    """

    contents = await cv_file.read()
    logger.info(cv_file.filename)

    try:
        if cv_file.filename.endswith(".pdf"):
            response = request.app.state.model_gen(contents)

        return JSONResponse(
            content={
                "filename": cv_file.filename,
                "info_extract": json.loads(response),
            }
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")
