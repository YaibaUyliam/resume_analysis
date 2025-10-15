import logging
import traceback
import re
import os
from elasticsearch import Elasticsearch, AsyncElasticsearch

from .manager import GenerationManager, EmbeddingManager
from .utils import convert_resume_format
from .providers.prompt.resume_prompt import PROMPT, SYSTEM, TASK

logger = logging.getLogger(__name__)


class ResumeService:
    def __init__(self):
        self.generation_manager = GenerationManager()
        self.embedding_manager = EmbeddingManager()

        self.es_client = AsyncElasticsearch(hosts=["http://localhost:9200"])
        self.index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.index_name}")

    async def _store_resume(self, gen_res, emb_res, cv_id):
        year_of_experience = gen_res["personal_info"]["year_of_experience"]
        if year_of_experience:
            match = re.search(r"\d+", str(year_of_experience))
            if match:
                year_of_experience = float(match.group(0))

        doc = {
            "id": cv_id,
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "year_of_experience": year_of_experience,
            "embedding_vector": emb_res,
        }
        logger.info(doc["keywords"])

        resp = await self.es_client.index(index=self.index_name, document=doc)
        logger.info(resp)
        await self.es_client.close()

    async def extract_and_store(self, contents, prompt, suffix, cv_id=None):
        model_gen = await self.generation_manager.init_model()
        model_emb = await self.embedding_manager.init_model()

        if prompt is None:
            prompt = PROMPT
        gen_res, resume_text = await model_gen(contents, prompt, SYSTEM, suffix)

        gen_res_format = convert_resume_format(gen_res)
        emb_res = await model_emb([resume_text], TASK)

        if cv_id:
            logger.info("Saving resume ....")
            try:
                await self._store_resume(gen_res, emb_res[0], cv_id)
            except:
                logger.info("Save data failed!!!!!!")
                logger.error(traceback.format_exc())

        return gen_res, gen_res_format
