from io import BytesIO
import base64
import tempfile
import re
import subprocess
import os
import multiprocessing
import requests

from pdf2image import convert_from_bytes
from PIL import Image
from markitdown import MarkItDown
from loguru import logger

from typing import Optional
from abc import ABC, abstractmethod

from app.agent.ocr_worker import run_ocr_process


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


def unload_model_ollama(model_name):
    ollama_url = os.environ.get("OLLAMA_BASE_URL")

    unload_model_resp = requests.post(
        (
            ollama_url + "/api/generate"
            if ollama_url
            else "http://0.0.0.0:11434/api/generate"
        ),
        json={"model": model_name, "keep_alive": 0},
    )

    if unload_model_resp.status_code == 200:
        logger.info(unload_model_resp.text)
    else:
        logger.info("Model already stopped")


class PreprocessData:
    def __init__(self):
        use_vision = int(os.environ.get("USE_VISION", 0))

        if use_vision == 1:
            self.use_vision = True
        else:
            self.use_vision = False
            self.md = MarkItDown(enable_plugins=False)

    def convert_data(self, data: bytes | str, file_suffix: str) -> str | list[str]:
        if self.use_vision:
            return convert_pdf_to_img_base64(data)

        if isinstance(data, str):
            return data

        if isinstance(data, bytes) and file_suffix == ".pdf":
            unload_model_ollama(os.environ["LL_MODEL"])

            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            p = ctx.Process(target=run_ocr_process, args=(data, queue))
            p.start()
            p.join()

            if not queue.empty():
                res = queue.get()
                if res["status"] == False:
                    logger.info(res["traceback"])
                    raise res["exception"]

                return res["data"]

        if isinstance(data, bytes) and file_suffix:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_suffix
            ) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                temp_path = temp_file.name

                if file_suffix == ".doc":
                    temp_path = convert_doc_to_docx(temp_path)

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
