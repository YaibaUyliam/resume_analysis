from io import BytesIO
import base64
import tempfile
import re

from pdf2image import convert_from_bytes
from PIL import Image
from markitdown import MarkItDown

from paddleocr import PaddleOCR
import numpy as np

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


def paddleocrv3_output_to_text(rec_polys, rec_texts, rec_scores):
    text_by_line = ""
    ocrOnly = {}

    for idx in range(len(rec_polys)):
        if rec_scores[idx] < 0.3:
            continue

        points = np.array(rec_polys[idx]).astype(np.int32).tolist()
        x1 = min(points[0][0], points[1][0], points[2][0], points[3][0])
        x2 = max(points[0][0], points[1][0], points[2][0], points[3][0])
        y1 = min(points[0][1], points[1][1], points[2][1], points[3][1])
        y2 = max(points[0][1], points[1][1], points[2][1], points[3][1])
        y_c = int((y1 + y2) / 2)

        if idx == 0:
            ocrOnly[0] = [[x1, y1, x2, y2]]
            text_by_line += rec_texts[idx]
            text_by_line += " "

        if idx > 0:
            sameLine = False
            for key in ocrOnly:
                for idxBb, bbox in enumerate(ocrOnly[key]):
                    x1_l, y1_l, x2_l, y2_l = ocrOnly[key][idxBb]
                    if y1_l < y_c < y2_l:
                        sameLine = True
                        ocrOnly[key].append([x1, y1, x2, y2])
                        text_by_line += rec_texts[idx]
                        text_by_line += " "

                    if sameLine:
                        break
                if sameLine:
                    break
            if sameLine == False:
                key = [key for key in ocrOnly][-1] + 1
                ocrOnly[key] = [[x1, y1, x2, y2]]
                text_by_line = text_by_line.strip()
                text_by_line += "\n"
                text_by_line += rec_texts[idx]
                text_by_line += " "

    return text_by_line.strip()


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

    def convert_data(self, data: bytes | str, file_suffix: str):
        if self.use_vision:
            return convert_pdf_to_img_base64(data)

        if isinstance(data, str):
            return data

        if isinstance(data, bytes) and file_suffix == ".pdf":
            ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="ch",
                text_detection_model_dir="./ckpts/PP-OCRv5_server_det",
                text_recognition_model_dir="./ckpts/PP-OCRv5_server_rec",
            )

            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
                temp_pdf.write(data)
                temp_pdf.flush()

                result = ocr.predict(temp_pdf.name)

            text_by_line = ""
            for res in result:
                convert = paddleocrv3_output_to_text(
                    res["rec_polys"], res["rec_texts"], res["rec_scores"]
                )
                text_by_line += convert + "\n\n"

            return text_by_line

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
    async def __call__(
        self, resume_data: bytes, prompt: Optional[str], file_suffix: str
    ) -> str: ...


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """

    def __init__(self):
        self.md = MarkItDown(enable_plugins=False)

    @abstractmethod
    async def __call__(self, resume_data: str, query: bool = False) -> list[float]: ...
