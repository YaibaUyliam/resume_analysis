from io import BytesIO
import base64
import tempfile
import re

from pdf2image import convert_from_bytes
from PIL import Image
from markitdown import MarkItDown

from typing import Optional
from abc import ABC, abstractmethod


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


class ExtractionProvider(ABC):
    """
    Abstract base class for providers.
    """

    def __init__(self, use_vision: int):
        self.use_vision = False

        if use_vision == 1:
            self.use_vision = True
        else:
            self.use_vision = False
            self.md = MarkItDown(enable_plugins=False)

    def convert_data(self, resume_data: bytes, file_suffix: str):
        if self.use_vision:
            return convert_pdf_to_img_base64(resume_data)

        else:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_suffix
            ) as temp_file:
                temp_file.write(resume_data)
                temp_file.flush()
                temp_path = temp_file.name

                return self.md.convert(temp_path).text_content

    @abstractmethod
    async def __call__(
        self, resume_data: bytes, prompt: Optional[str], file_suffix: str
    ) -> str: ...


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    @abstractmethod
    async def __call__(self, text: str) -> list[float]: ...
