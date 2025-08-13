import logging
from io import BytesIO
import base64
import re

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from pdf2image import convert_from_bytes
from PIL import Image
from fastapi.concurrency import run_in_threadpool

from .qwen_vl_utils import process_vision_info


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


def create_chat_template(base64_imgs: str, promtp: str):
    if not base64_imgs:
        logger.error("Not get base64 images")
        return None

    messages = []
    content = []

    for base64_img in base64_imgs:
        content.append({"image_url": f"data:image/jpeg;base64,{base64_img}"})
        # content.append({"image": f"file://{q[0]}"})

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
    def __init__(self, model_path: str, torch_dtype: str):
        logger.info("Running model with HugingFace ........")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except:
            logger.warning("flash_attention_2 is not used !!!")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

        logger.info("Model and processor loaded!")

    def _preprocess_data(self, resume_data: bytes, prompt: str):
        imgs = convert_pdf_to_img_base64(resume_data)

        logger.info(len(imgs))

        messages = create_chat_template(imgs, prompt)
        # if messages:
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

    def _predict(self, resume_data: bytes, prompt: str):
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

    async def __call__(self, resume_data: bytes, prompt: str):
        res = await run_in_threadpool(self._predict, resume_data, prompt)
        return res[0]
