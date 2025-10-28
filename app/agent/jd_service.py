import logging
import traceback
import re
import os
from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, AsyncElasticsearch

from .manager import GenerationManager, EmbeddingManager
from .utils import convert_jd_format
from .providers.prompt.jd_prompt import PROMPT, SYSTEM, TASK
from .providers.prompt.resume_review import PROMPT_REVIEW, SYSTEM_REVIEW


logger = logging.getLogger(__name__)


class JDService:
    def __init__(self):
        self.generation_manager = GenerationManager()
        self.embedding_manager = EmbeddingManager()

        self.es_client = AsyncElasticsearch(hosts=[os.environ["ES_HOST"]])
        self.jd_index_name = os.environ["ES_JD_INDEX"]
        self.cv_index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.jd_index_name}")

        self.timezone = timezone(timedelta(hours=8))

    async def _store_resume(self, gen_res, emb_res, file_name, jd_id, jd_text):
        minimum_years_of_experience = gen_res["minimum_years_of_experience"]
        if minimum_years_of_experience:
            match = re.search(r"\d+", str(minimum_years_of_experience))
            if match:
                minimum_years_of_experience = float(match.group(0))

        doc = {
            "id": jd_id,
            "jd_url": file_name,
            "content": jd_text,
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "job_name": gen_res["job_name"],
            "job_description": gen_res["job_description"],
            "minimum_years_of_experience": minimum_years_of_experience,
            "required_skills": gen_res["required_skills"],
            "embedding_vector": emb_res,
            "created_at": datetime.now(self.timezone).isoformat(),
        }

        resp = await self.es_client.index(index=self.jd_index_name, document=doc)
        logger.info(resp)
        await self.es_client.close()

    async def _keywords_search(self, keywords):
        query = {
            "_source": {"excludes": ["embedding_vector"]},
            "size": 10,
            "query": {"match": {"keywords": keywords}},
        }
        response = await self.es_client.search(index=self.cv_index_name, body=query)

        return response["hits"]["hits"]

    async def _vectors_search(self, query_vector: list):
        response = await self.es_client.search(
            index=self.cv_index_name,
            body={
                "_source": {"excludes": ["embedding_vector"]},
                "size": 10,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                            "params": {"query_vector": query_vector},
                        },
                    }
                },
            },
        )

        return response["hits"]["hits"]

    async def match(self, keywords, vector):
        keywords_search_res = await self._keywords_search(keywords)
        vectors_search_res = await self._vectors_search(vector)
        logger.info(keywords_search_res)

        cv_list = []
        for hit in vectors_search_res:
            cv_list.append(hit)

        return cv_list

    async def review(self, model_gen, jd_content, jd_keywords, cv_list: list[dict]):
        results = []

        for resume in cv_list:
            prompt = PROMPT_REVIEW.format(
                raw_job_description=jd_content,
                extracted_job_keywords=jd_keywords,
                raw_resume=resume["content"],
                extracted_resume_keywords=resume["keywords"],
            )

            gen_res, _ = await model_gen("", prompt, SYSTEM_REVIEW, None)
            results.append({**resume, **gen_res})

        return results

    async def _pre_data(self, contents: dict):
        key_rm = [
            "fromDate",
            "toDate",
            "departmentId",
            "category",
            "types",
            "benefits",
            "locations",
            "isActive",
            "isDeleted",
            "createdAt",
            "updatedAt",
            "createdBy",
            "createdByUsername",
            "updatedBy",
            "updatedByUsername",
            "_class",
        ]

        dict2str = "\n".join(
            [f"{k}: {v}" for k, v in contents.items() if k not in key_rm]
        )
        return dict2str

    async def extract_match_review(
        self, contents: bytes | dict, prompt, file_name, jd_id=None
    ):
        if isinstance(contents, dict):
            contents = await self._pre_data(contents)

        model_gen = await self.generation_manager.init_model()
        model_emb = await self.embedding_manager.init_model()

        if prompt is None:
            prompt = PROMPT
        suffix = "." + file_name.split(".")[-1]
        gen_res, jd_text = await model_gen(contents, prompt, SYSTEM, suffix)

        # gen_res_format = convert_jd_format(gen_res)
        emb_res = await model_emb([jd_text], TASK, query=True)

        if gen_res["extracted_keywords"]:
            cv_matcher = await self.match(
                keywords=", ".join(gen_res["extracted_keywords"]), vector=emb_res[0]
            )

            cv_top_k_review = await self.review(
                model_gen, contents, gen_res["extracted_keywords"], cv_matcher
            )

            if jd_id:
                logger.info("Saving resume ....")
                try:
                    await self._store_resume(
                        gen_res, emb_res[0], file_name, jd_id, jd_text
                    )
                except:
                    logger.info("Save data failed!!!!!!")
                    logger.error(traceback.format_exc())

        else:
            logger.info("Can not get extracted keywords")

        await self.es_client.close()
        return gen_res, cv_top_k_review


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
