from __future__ import annotations

from io import BytesIO
import base64
import tempfile
import re
import subprocess
import os
import multiprocessing

from pdf2image import convert_from_bytes
from PIL import Image
from loguru import logger

from typing import Optional
from abc import ABC, abstractmethod


def get_markitdown_class():
    try:
        from markitdown import MarkItDown

        return MarkItDown
    except (ImportError, AttributeError):
        pass

    try:
        from markitdown._markitdown import MarkItDown

        return MarkItDown
    except (ImportError, AttributeError):
        pass

    raise ImportError(
        "MarkItDown is not available in the installed 'markitdown' package. "
        "Please install a compatible version or use PDF/vision mode only."
    )


def extract_pdf_text_with_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF text fallback extraction."
        ) from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            texts.append(page_text)

    extracted_text = "\n\n".join(texts).strip()
    if not extracted_text:
        raise ValueError("No text could be extracted from the PDF via pypdf.")
    return extracted_text

def remove_image_special(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    ch_special = ["<ref>", "<ref>", "```", "json"]
    for ch in ch_special:
        text = text.replace(ch, "")

    text = text.strip()

    return re.sub(r"<box>.*?(</box>|$)", "", text)


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


def convert_doc_to_docx(input_file):
    input_path = os.path.abspath(input_file)
    output_dir = os.path.dirname(input_path)

    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--invisible",
            "--norestore",
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            "--convert-to",
            "docx",
            "--outdir",
            output_dir,
            input_path,
        ],
        check=True,
    )

    return input_path.replace(".doc", ".docx")


class PreprocessData:
    def __init__(self):
        use_vision = int(os.environ.get("USE_VISION", 0))
        self.md = None

        if use_vision == 1:
            self.use_vision = True
        else:
            self.use_vision = False
            try:
                MarkItDown = get_markitdown_class()
                self.md = MarkItDown(enable_plugins=False)
            except ImportError as exc:
                logger.warning(str(exc))

    def convert_data(self, data: bytes | str, file_suffix: str) -> str | list[str]:
        if self.use_vision:
            return convert_pdf_to_img_base64(data)

        if isinstance(data, str):
            return data

        if isinstance(data, bytes) and file_suffix == ".pdf":
            try:
                from app.agent.ocr_worker import run_ocr_process

                subprocess.run(
                    ["ollama", "stop", os.environ.get("LL_MODEL")],
                    capture_output=True,
                    text=True,
                )

                ctx = multiprocessing.get_context("spawn")
                queue = ctx.Queue()
                p = ctx.Process(target=run_ocr_process, args=(data, queue))
                p.start()
                p.join()

                if not queue.empty():
                    res = queue.get()
                    if res["status"] is False:
                        logger.info(res["traceback"])
                        raise res["exception"]

                    return res["data"]
            except Exception as exc:
                logger.warning(
                    f"OCR pipeline failed for PDF extraction: {exc}. "
                    "Falling back to pypdf text extraction."
                )
                return extract_pdf_text_with_pypdf(data)

        if isinstance(data, bytes) and file_suffix:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_suffix
            ) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                temp_path = temp_file.name

                if file_suffix == ".doc":
                    temp_path = convert_doc_to_docx(temp_path)

                if self.md is None:
                    raise ImportError(
                        "MarkItDown is required to convert non-PDF documents. "
                        "Install a compatible 'markitdown' package or enable vision mode."
                    )
                return self.md.convert(temp_path).text_content

        raise TypeError("resume_data is not valid type")


class ExtractionProvider(ABC):
    """
    Abstract base class for providers.
    """

    def __init__(self, use_vision: bool):
        self.use_vision = use_vision

    @abstractmethod
    async def __call__(
        self, resume_data: bytes, prompt: Optional[str], file_suffix: str
    ) -> str: ...


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    def __init__(self):
        pass

    @abstractmethod
    async def __call__(self, resume_data: str, query: bool = False) -> list[float]: ...
