from io import BytesIO
import base64
import tempfile
import re
import fitz

from pdf2image import convert_from_bytes
from PIL import Image
from markitdown import MarkItDown
from typing import Optional
from abc import ABC, abstractmethod
from paddleocr import PaddleOCR
from threading import Lock


class PaddleOCRSingleton:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, lang="ch"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang,
                        show_log=False
                    )
        return cls._instance

def remove_pdf_watermark(
    pdf_bytes: bytes,
    watermark_patterns: list[str] | None = None
) -> bytes:
    """
    Remove text-based watermark from PDF.
    watermark_patterns: regex patterns to remove
    """
    if not watermark_patterns:
        watermark_patterns = [
            r"confidential",
            r"draft",
            r"sample",
            r"downloaded from.*",
            r"page \d+",
        ]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4]
            if not text:
                continue

            for pattern in watermark_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    rect = fitz.Rect(block[:4])
                    page.add_redact_annot(rect, fill=(1, 1, 1))

        page.apply_redactions()

    output = BytesIO()
    doc.save(output)
    doc.close()

    return output.getvalue()



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

    def __init__(self, mode: str = "md", ocr_lang: str= "ch"):
        self.mode = mode

        if mode == "vision":
            self.use_vision = True
        elif mode == "paddleocr":
            self.use_vision = False
            self.ocr = PaddleOCRSingleton.get_instance(lang=ocr_lang)
        else:
            self.use_vision = False
            self.md = MarkItDown(enable_plugins=False)

    def pdf_to_text_paddleocr(self, pdf_bytes: bytes) -> str:
        images = convert_from_bytes(pdf_bytes)

        results = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f.name, "JPEG")
                ocr_result = self.ocr.ocr(f.name, cls=True)

            for page in ocr_result:
                for line in page:
                    results.append(line[1][0])

        return "\n".join(results)

    def convert_data(self, data: bytes|str, file_suffix: str):
        # Vision mode
        if self.use_vision:
            return convert_pdf_to_img_base64(data)
        
        # PaddleOCR mode
        if self.mode == "paddleocr":
            if not isinstance(data, bytes):
                raise TypeError("PaddleOCR requires pdf bytes")
            return self.pdf_to_text_paddleocr(data)
        
        #Markitdown mode
        if isinstance(data, str):
            return data

        if isinstance(data, bytes) and file_suffix:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_suffix
            ) as temp_file:
                temp_file.write(data)
                temp_file.flush()
                temp_path = temp_file.name

                return self.md.convert(temp_path).text_content
            
        raise TypeError("resume_data is not valid type")

    @abstractmethod
    # async def __call__(
    #     self, resume_data: bytes, prompt: Optional[str], file_suffix: str
    # ) -> str: ...

    async def __call__(
        self,
        resume_data: bytes,
        prompt: Optional[str],
        file_suffix: str
    ) -> str:
        if isinstance(resume_data, bytes) and file_suffix.lower() == ".pdf":
            resume_data = remove_pdf_watermark(resume_data)

        result = self.convert_data(resume_data, file_suffix)
        
        if isinstance(result, str):
            return remove_image_special(result)

        return result


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    def __init__(self):
        self.md = MarkItDown(enable_plugins=False)

    @abstractmethod
    async def __call__(self, resume_data: str, query: bool = False) -> list[float]: ...
