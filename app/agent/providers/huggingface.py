import logging
import re

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from fastapi.concurrency import run_in_threadpool
from typing import Optional

from .qwen_vl_utils import process_vision_info
from .prompt import PROMPT
from .base import ExtractionProvider, EmbeddingProvider, remove_image_special


logger = logging.getLogger(__name__)


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


class TorchExtractionProvider(ExtractionProvider):
    def __init__(self, model_path: str, torch_dtype: str, use_vision: int):
        logger.info("Running model with HugingFace ........")
        super().__init__(use_vision)

        max_memory = {
            0: "14GiB",  # GPU 0 cho phép dùng nhiều hơn
            1: "3GiB",  # GPU 1 ít hơn
        }
        # try:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            # attn_implementation="eager",
            device_map="cuda:0",
        ).eval()

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

        logger.info("Model and processor loaded!")

    def _preprocess_data(self, resume_data: bytes, prompt: str):
        converted_data = self.convert_data(resume_data)

        logger.info(len(converted_data))

        messages = create_chat_template(converted_data, prompt)
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

    def _predict(self, resume_data: bytes, prompt: Optional[str]):
        if not prompt:
            prompt = PROMPT

        processed_resume_data = self._preprocess_data(resume_data, prompt)

        inputs = processed_resume_data.to(self.model.device)

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

        output_text = [remove_image_special(v) for v in output_text]

        return output_text

    async def __call__(self, resume_data: bytes, prompt: Optional[str]):
        res = await run_in_threadpool(self._predict, resume_data, prompt)
        return res[0]
