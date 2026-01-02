import logging
import traceback
import re
import os

from loguru import logger
from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, AsyncElasticsearch

from .manager import GenerationManager, EmbeddingManager
from .utils import convert_resume_format
from .providers.prompt.resume_prompt import PROMPT, SYSTEM, TASK

# logger = logging.getLogger(__name__)


class ResumeService:
    def __init__(self):
        self.generation_manager = GenerationManager()
        self.embedding_manager = EmbeddingManager()

        self.es_client = AsyncElasticsearch(hosts=[os.environ["ES_HOST"]])
        self.index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.index_name}")

        self.timezone = timezone(timedelta(hours=8))

    async def _store_resume(self, gen_res, emb_res, file_name, cv_id, resume_text):
        year_of_experience = gen_res["personal_info"]["year_of_experience"]
        if year_of_experience:
            match = re.search(r"\d+", str(year_of_experience))
            if match:
                year_of_experience = float(match.group(0))

        doc = {
            "id": cv_id,
            "cv_url": file_name,
            "content": resume_text,
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "year_of_experience": year_of_experience,
            "embedding_vector": emb_res,
            "full_name": gen_res["personal_info"]["full_name"],
            "desired_position": gen_res["personal_info"].get("desired_position"),
            "created_at": datetime.now(self.timezone).isoformat(),
        }

        resp = await self.es_client.index(index=self.index_name, document=doc)
        logger.info(resp)
        await self.es_client.close()

    async def extract_and_store(self, contents, sys_mess, file_name, cv_id=None):
        model_gen = await self.generation_manager.init_model()
        model_emb = await self.embedding_manager.init_model()

        if sys_mess is None:
            sys_mess = SYSTEM

        suffix = "." + file_name.split(".")[-1]
        gen_res, resume_text = await model_gen(contents, PROMPT, sys_mess, suffix)
        # logger.info(gen_res)
        gen_res_format = convert_resume_format(gen_res)

        emb_res = await model_emb([resume_text], TASK)

        if cv_id:
            logger.info("Saving resume ....")
            try:
                await self._store_resume(gen_res, emb_res[0], file_name, cv_id, resume_text)
            except:
                logger.info("Save data failed!!!!!!")
                logger.error(traceback.format_exc())

        return gen_res, gen_res_format
