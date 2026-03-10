import traceback
import re
import os
import subprocess

from loguru import logger
from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, AsyncElasticsearch
from pydantic import BaseModel, Field
from typing import List

from app.agent.utils import convert_resume_format
from app.agent.providers import (
    PreprocessData,
    OllamaExtractionProvider,
    OllamaEmbeddingProvider,
)
from app.agent.providers.prompt.resume_prompt import PROMPT, SYSTEM, TASK


class ResumeSchema(BaseModel):
    id: str
    cv_url: str
    content: str
    keywords: str | None
    year_of_experience: float | None
    embedding_vector: List[float]
    full_name: str | None
    desired_position: str | None
    created_at: str
    updated_at: str
    is_deleted: bool = False


class ResumeDuplicate(BaseModel):
    cv_id: str
    cv_url: str
    duplicated_cv: list[dict] = []
    # is_duplicate: bool = False
    # cv_id_duplicates: list = []
    # cv_url_duplicates: list = []
    # similar_percentage: list = []


class ResumeService:
    def __init__(self):
        self.preprocess_data = PreprocessData()

        model_extract_name = os.environ["LL_MODEL"]
        self.model_extract = OllamaExtractionProvider(model_extract_name)
        model_embed_name = os.environ["EMBEDDING_MODEL"]
        self.model_embed = OllamaEmbeddingProvider(model_embed_name)
        self.similar_thresh = float(os.environ["SIMILAR_THRESH"])

        self.es_client = AsyncElasticsearch(hosts=[os.environ["ES_HOST"]])
        self.index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.index_name}")

        self.timezone = timezone(timedelta(hours=8))

    async def _store_resume(self, gen_res, emb_res, file_name, cv_id, resume_data):
        year_of_experience = gen_res["personal_info"]["year_of_experience"]
        if year_of_experience:
            match = re.search(r"\d+", str(year_of_experience))
            if match:
                year_of_experience = float(match.group(0))

        time_current = datetime.now(self.timezone).isoformat()
        doc = {
            "id": cv_id,
            "cv_url": file_name,
            "content": resume_data if isinstance(resume_data, str) else "",
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "year_of_experience": year_of_experience if year_of_experience else None,
            "embedding_vector": emb_res,
            "full_name": gen_res["personal_info"]["full_name"],
            "desired_position": gen_res["personal_info"].get("desired_position"),
            "created_at": time_current,
            "updated_at": time_current,
        }

        resume_extract_result = ResumeSchema(**doc)

        resp = await self.es_client.index(
            index=self.index_name, document=resume_extract_result.model_dump()
        )
        logger.info(resp)

    async def del_cv(self, cv_id):
        search_resp = await self.es_client.search(
            index=self.index_name,
            query={"term": {"id": cv_id}},
            source=["_id"],
        )

        hits = search_resp["hits"]["hits"]
        if not hits:
            logger.info("CV not exist !!!!")
            return None

        es_id = hits[0]["_id"]
        logger.info(f"{es_id} will be deleted !!!")
        doc = {
            "is_deleted": True,
            "updated_at": datetime.now(self.timezone).isoformat(),
        }
        resp = await self.es_client.update(index=self.index_name, id=es_id, doc=doc)

        return resp

    async def close(self):
        logger.info("close connection to ES")
        await self.es_client.close()

    async def _vectors_search(self, query_vector: list, size=1):
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "_source": {"excludes": ["embedding_vector"]},
                "size": size,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {"must_not": [{"term": {"is_deleted": True}}]}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                            "params": {"query_vector": query_vector},
                        },
                    }
                },
            },
        )

        return response["hits"]["hits"]

    async def check_duplication(
        self, data, file_name
    ) -> tuple[list, str | list[str], list[float]]:
        try:
            suffix = "." + file_name.split(".")[-1]
            data_converted = self.preprocess_data.convert_data(data, suffix)
            emb_result = await self.model_embed([data_converted], TASK)
            emb_result = emb_result[0]
            embed_query_result = await self._vectors_search(emb_result, size=2)

            check_result = []
            for cv_info in embed_query_result:
                raw_score = cv_info["_score"]
                cosine_sim = raw_score - 1.0
                similar_percentage = round(max(0.0, cosine_sim), 2)
                logger.info(f"Similar Percentage: {similar_percentage}")

                if similar_percentage > self.similar_thresh:
                    cv_similar = {
                        "cv_id": cv_info["_source"]["id"],
                        "cv_url": cv_info["_source"]["cv_url"],
                        "similar_percentage": similar_percentage,
                    }
                    check_result.append(cv_similar)

            logger.info(check_result)

            return check_result, data_converted, emb_result

        except ValueError as e:
            logger.error("OCR failed!!!!!")
            raise

        except Exception as e:
            logger.error(e)
            raise

    async def extract(self, data, sys_mess, file_name):
        if sys_mess is None:
            sys_mess = SYSTEM
        suffix = "." + file_name.split(".")[-1]
        data_converted = self.preprocess_data.convert_data(data, suffix)

        gen_res, service_resp = await self.model_extract(
            data_converted, PROMPT, sys_mess
        )
        # logger.info(gen_res)
        gen_res_format = convert_resume_format(gen_res)

        return gen_res, gen_res_format, service_resp

    # Already convert data in step check duplication
    async def extract_and_store(self, data, file_name, cv_id, cv_embed):
        try:
            sys_mess = SYSTEM
            gen_res, _ = await self.model_extract(data, PROMPT, sys_mess)
            gen_res_format = convert_resume_format(gen_res)

            logger.info("Saving resume ....")
            await self._store_resume(gen_res, cv_embed, file_name, cv_id, data)

        except:
            logger.info("Save data failed!!!!!!")
            logger.error(traceback.format_exc())
            raise

        return gen_res, gen_res_format


_resume_service_instance = None


def get_resume_service() -> ResumeService:
    global _resume_service_instance
    if _resume_service_instance is None:
        logger.info("Initing Resume Service ....")
        _resume_service_instance = ResumeService()
    return _resume_service_instance
