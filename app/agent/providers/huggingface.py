import logging
from io import BytesIO
import base64
import re
import tempfile

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from pdf2image import convert_from_bytes
from PIL import Image
from fastapi.concurrency import run_in_threadpool
from typing import Optional
from markitdown import MarkItDown

from .qwen_vl_utils import process_vision_info
from .prompt import PROMPT


logger = logging.getLogger(__name__)


def _remove_image_special(text):
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


def create_chat_template(data_convert: str | list[str], promtp: str):
    messages = []
    content = []

    if isinstance(data_convert, list):
        for base64_img in data_convert:
            content.append({"image_url": f"data:image/jpeg;base64,{base64_img}"})
    else:
        content.append({"text": data_convert})
        logger.info(content)

    content.append({"text": promtp})
    messages.append({"role": "user", "content": content})

    return messages


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message["content"]:
            if "image" in item:
                new_item = {"type": "image", "image": item["image"]}
            elif "text" in item:
                new_item = {"type": "text", "text": item["text"]}
            elif "video" in item:
                new_item = {"type": "video", "video": item["video"]}
            elif "image_url" in item:
                new_item = {"type": "image_url", "image_url": item["image_url"]}
            else:
                continue
            new_content.append(new_item)

        new_message = {"role": message["role"], "content": new_content}
        transformed_messages.append(new_message)

    return transformed_messages


class TorchExtractionProvider:
    def __init__(self, model_path: str, torch_dtype: str, use_vision: int):
        logger.info("Running model with HugingFace ........")

        max_memory = {
            0: "14GiB",  # GPU 0 cho phép dùng nhiều hơn
            1: "3GiB"   # GPU 1 ít hơn
        }
        # try:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            # attn_implementation="eager",
            device_map="cuda:0",
            # device_map="auto",
            # max_memory=max_memory
        ).eval()

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

        if use_vision == 1:
            logger.info("Use model vision to extract images")
            self.use_vision = True
        else:
            self.use_vision = False
            self.md = MarkItDown(enable_plugins=False)

        logger.info("Model and processor loaded!")

    def _preprocess_data(self, resume_data: bytes, prompt: Optional[str] = None):
        if self.use_vision:
            data_convert = convert_pdf_to_img_base64(resume_data)

        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(resume_data)
                temp_path = temp_file.name

            data_convert = self.md.convert(temp_path).text_content

        logger.info(len(data_convert))

        messages = create_chat_template(data_convert, prompt)
        messages = _transform_messages(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def _predict(self, resume_data: bytes, prompt: Optional[str] = None):
        if not prompt:
            prompt = PROMPT

        resume_data_processed = self._preprocess_data(resume_data, prompt)

        inputs = resume_data_processed.to(self.model.device)

        # same prompt, same result if do_sample=False
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=5000, do_sample=True
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output_text = [_remove_image_special(v) for v in output_text]

        return output_text

    async def __call__(self, resume_data: bytes, prompt: Optional[str] = None):
        res = await run_in_threadpool(self._predict, resume_data, prompt)
        return res[0]
