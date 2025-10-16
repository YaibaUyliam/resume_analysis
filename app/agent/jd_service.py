import logging
import traceback
import re
import os
from elasticsearch import Elasticsearch, AsyncElasticsearch

from .manager import GenerationManager, EmbeddingManager
from .utils import convert_jd_format
from .providers.prompt.jd_prompt import PROMPT, SYSTEM, TASK

logger = logging.getLogger(__name__)


class JDService:
    def __init__(self):
        self.generation_manager = GenerationManager()
        self.embedding_manager = EmbeddingManager()

        self.es_client = AsyncElasticsearch(hosts=[os.environ["ES_HOST"]])
        self.jd_index_name = os.environ["ES_JD_INDEX"]
        self.cv_index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.jd_index_name}")

    async def _store_resume(self, gen_res, emb_res, jd_id):
        minimum_years_of_experience = gen_res["minimum_years_of_experience"]
        if minimum_years_of_experience:
            match = re.search(r"\d+", str(minimum_years_of_experience))
            if match:
                minimum_years_of_experience = float(match.group(0))

        doc = {
            "id": jd_id,
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "job_name": gen_res["job_name"],
            "job_description": gen_res["job_description"],
            "minimum_years_of_experience": minimum_years_of_experience,
            "required_skills": gen_res["required_skills"],
            "embedding_vector": emb_res,
        }

        resp = await self.es_client.index(index=self.index_name, document=doc)
        logger.info(resp)
        await self.es_client.close()

    async def match(self, keywords):
        query = {
            "_source": {"excludes": ["embedding_vector"]},
            "size": 10,
            "query": {"match": {"keywords": keywords}},
        }
        logger.info(query)
        response = await self.es_client.search(index=self.cv_index_name, body=query)

        cv_list = []
        for hit in response["hits"]["hits"]:
            cv_list.append(hit)

        return cv_list

    async def extract_and_match(self, contents, prompt, suffix, jd_id=None):
        model_gen = await self.generation_manager.init_model()
        model_emb = await self.embedding_manager.init_model()

        if prompt is None:
            prompt = PROMPT
        gen_res, jd_text = await model_gen(contents, prompt, SYSTEM, suffix)

        # gen_res_format = convert_jd_format(gen_res)
        emb_res = await model_emb([jd_text], TASK, query=True)

        if gen_res["extracted_keywords"]:
            cv_matcher = await self.match(", ".join(gen_res["extracted_keywords"]))
            logger.info(cv_matcher)

        if jd_id:
            logger.info("Saving resume ....")
            try:
                await self._store_resume(gen_res, emb_res[0], jd_id)
            except:
                logger.info("Save data failed!!!!!!")
                logger.error(traceback.format_exc())

        await self.es_client.close()
        return gen_res, cv_matcher


if __name__ == "__main__":
    import asyncio
    import numpy as np

    service = JDService()
    print(service.cv_index_name)
    query = {
        "_source": {"excludes": ["embedding_vector"]},
        "size": 5,
        # "explain": False,
        "query": {
            "match": {
                "keywords": "Manual test, automation Testing, test case design, developing test plans"
            }
        },
    }
    res = asyncio.run(service.match(query=query))
    print(res)
    asyncio.run(service.es_client.close())
