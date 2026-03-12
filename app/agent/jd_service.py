from __future__ import annotations

import asyncio
import traceback
import re
import os

from loguru import logger
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, AsyncElasticsearch, ConflictError, NotFoundError

from app.agent.providers import (
    PreprocessData,
    OllamaExtractionProvider,
    OllamaEmbeddingProvider,
)
from app.agent.utils import convert_jd_format
from .providers.prompt.jd_prompt import PROMPT, SYSTEM, TASK
from .providers.prompt.resume_review import PROMPT_REVIEW, SYSTEM_REVIEW
from app.core.setting import settings

RRF_K = 60
RRF_CANDIDATE_MULTIPLIER = 3


@dataclass
class MatcherData:
    cv_id: str
    cv_url: str
    content: str
    year_of_experience: str
    keywords: str
    full_name: str

    match_score: Optional[int] = None
    strong_matches: Optional[List[str]] = None
    partial_matches: Optional[List[str]] = None
    missing_keywords: Optional[List[str]] = None
    review: Optional[str] = None

    @classmethod
    def get_cv_data_from_search_by_keyword(cls, v: dict):
        source = v["_source"]

        return cls(
            cv_id=source["id"],
            cv_url=source["cv_url"],
            content=source["content"],
            year_of_experience=source.get("year_of_experience"),
            full_name=source["full_name"],
            keywords=source["keywords"],
        )

    def merge_model_result_and_cv_data_original(self, gen_res: dict):
        self.match_score = gen_res["match_score"]
        self.strong_matches = gen_res["strong_matches"]
        self.partial_matches = gen_res["partial_matches"]
        self.missing_keywords = gen_res["missing_keywords"]
        self.review = gen_res["summary"]

    @classmethod
    def get_cv_info_from_search_in_past(cls, v: dict, search_result: dict, idx: int):
        source = v["_source"]

        return cls(
            cv_id=source["id"],
            cv_url=source["cv_url"],
            content=source["content"],
            year_of_experience=source.get("year_of_experience"),
            full_name=source["full_name"],
            keywords=source["keywords"],
            match_score=search_result["scores"][idx],
            strong_matches=search_result["strong_matches"][idx],
            partial_matches=search_result["partial_matches"][idx],
            missing_keywords=search_result["missing_keywords"][idx],
            review=search_result["summary"][idx],
        )


class JDService:
    def __init__(self):
        self.preprocess_data = PreprocessData()

        model_gen_name = os.environ["LL_MODEL"]
        self.model_gen = OllamaExtractionProvider(model_gen_name)
        model_embed_name = os.environ["EMBEDDING_MODEL"]
        self.model_embed = OllamaEmbeddingProvider(model_embed_name)

        self.es_client = AsyncElasticsearch(hosts=[settings.ES_HOST])
        self.jd_index_name = settings.ES_JD_INDEX
        self.search_result_index_name = settings.ES_SEARCH_RESULT_INDEX
        self.cv_index_name = settings.ES_CV_INDEX
        self._jd_index_ready = False
        self._search_result_index_ready = False
        logger.info(f"Index name: {self.jd_index_name, self.cv_index_name}")

        self.timezone = timezone(timedelta(hours=8))

    def _active_filters(self) -> list[dict]:
        return [
            {"bool": {"must_not": [{"term": {"is_deleted": True}}]}},
            {
                "bool": {
                    "should": [
                        {"term": {"is_active": True}},
                        {"bool": {"must_not": [{"exists": {"field": "is_active"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]

    def _hit_key(self, hit: dict) -> str | None:
        source = hit.get("_source", {})
        return hit.get("_id") or source.get("id")

    def _rrf_fuse_hits(
        self, hit_lists: list[list[dict]], size: int, rrf_k: int = RRF_K
    ) -> list[dict]:
        fused: dict[str, dict[str, Any]] = {}

        for hit_list in hit_lists:
            for rank, hit in enumerate(hit_list, start=1):
                hit_key = self._hit_key(hit)
                if not hit_key:
                    continue

                score = 1.0 / (rrf_k + rank)
                current = fused.get(hit_key)
                if current is None:
                    fused[hit_key] = {"score": score, "hit": hit}
                    continue

                current["score"] += score
                if hit.get("_score", 0.0) > current["hit"].get("_score", 0.0):
                    current["hit"] = hit

        ranked_hits = sorted(
            fused.values(), key=lambda item: item["score"], reverse=True
        )
        results = []
        for item in ranked_hits[:size]:
            hit = dict(item["hit"])
            hit["_score"] = item["score"]
            hit["rrf_score"] = item["score"]
            results.append(hit)
        return results

    def _jd_index_mapping(self, embedding_dims: int) -> dict:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "jd_url": {"type": "keyword"},
                    "content": {"type": "text"},
                    "keywords": {"type": "text"},
                    "job_name": {"type": "text"},
                    "job_description": {"type": "text"},
                    "minimum_years_of_experience": {"type": "float"},
                    "required_skills": {"type": "text"},
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": False,
                    },
                    "created_at": {"type": "date"},
                }
            },
        }

    def _search_result_index_mapping(self) -> dict:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "jd_id": {"type": "keyword"},
                    "top_cv_id": {"type": "keyword"},
                    "scores": {"type": "float"},
                    "strong_matches": {"type": "keyword"},
                    "partial_matches": {"type": "keyword"},
                    "missing_keywords": {"type": "keyword"},
                    "summary": {"type": "text"},
                }
            },
        }

    async def _ensure_jd_index(self, embedding_dims: int) -> None:
        if self._jd_index_ready:
            return

        exists = await self.es_client.indices.exists(index=self.jd_index_name)
        if exists:
            self._jd_index_ready = True
            return

        try:
            await self.es_client.indices.create(
                index=self.jd_index_name,
                body=self._jd_index_mapping(embedding_dims),
            )
            logger.info(
                f"Created Elasticsearch index {self.jd_index_name} with dims={embedding_dims}"
            )
        except ConflictError:
            logger.info(f"Elasticsearch index {self.jd_index_name} already exists")

        self._jd_index_ready = True

    async def _ensure_search_result_index(self) -> None:
        if self._search_result_index_ready:
            return

        exists = await self.es_client.indices.exists(index=self.search_result_index_name)
        if exists:
            self._search_result_index_ready = True
            return

        try:
            await self.es_client.indices.create(
                index=self.search_result_index_name,
                body=self._search_result_index_mapping(),
            )
            logger.info(
                f"Created Elasticsearch index {self.search_result_index_name}"
            )
        except ConflictError:
            logger.info(
                f"Elasticsearch index {self.search_result_index_name} already exists"
            )

        self._search_result_index_ready = True

    async def _store_jd(self, gen_res, emb_res, file_name, jd_id, jd_data):
        await self._ensure_jd_index(len(emb_res))
        minimum_years_of_experience = gen_res.get("minimum_years_of_experience", "")
        if minimum_years_of_experience:
            match = re.search(r"\d+", str(minimum_years_of_experience))
            if match:
                minimum_years_of_experience = float(match.group(0))

        doc = {
            "id": jd_id,
            "jd_url": file_name,
            "content": jd_data if isinstance(jd_data, str) else "",
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "job_name": gen_res.get("job_name", ""),
            "job_description": gen_res.get("job_description", ""),
            "minimum_years_of_experience": minimum_years_of_experience,
            "required_skills": gen_res.get("required_skills", ""),
            "embedding_vector": emb_res,
            "created_at": datetime.now(self.timezone).isoformat(),
        }

        resp = await self.es_client.index(index=self.jd_index_name, document=doc)
        logger.info(resp)

    async def _store_search_result(
        self, jd_id: str, cv_top_k_review: list[MatcherData]
    ):
        await self._ensure_search_result_index()
        top_cv_id = []
        scores = []
        strong_matches = []
        partial_matches = []
        missing_keywords = []
        summary = []
        for v in cv_top_k_review:
            scores.append(v.match_score)
            top_cv_id.append(v.cv_id)
            strong_matches.append(v.strong_matches)
            partial_matches.append(v.partial_matches)
            missing_keywords.append(v.missing_keywords)
            summary.append(v.review)

        doc = {
            "jd_id": jd_id,
            "top_cv_id": top_cv_id,
            "scores": scores,
            "strong_matches": strong_matches,
            "partial_matches": partial_matches,
            "missing_keywords": missing_keywords,
            "summary": summary,
        }
        resp = await self.es_client.index(index=self.search_result_index_name, document=doc, id=jd_id) # fmt:skip
        logger.info(resp)

    async def _get_search_result(self, jd_id):
        query = {"query": {"match": {"jd_id": jd_id}}}
        try:
            response = await self.es_client.search(
                index=self.search_result_index_name, body=query
            )
        except NotFoundError:
            logger.info(f"Index {self.search_result_index_name} does not exist")
            return []

        return response["hits"]["hits"]

    async def _keywords_search(self, keywords, job_name, size=5):
        query_content = job_name + keywords
        logger.info(f"Query content: {query_content}")

        query = {
            "_source": {"excludes": ["embedding_vector"]},
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_content,
                                "fields": [
                                    "keywords",
                                    "desired_position^2",
                                    "full_name^1.5",
                                    "canonical_cv_text",
                                    "content",
                                ],
                            }
                        }
                    ],
                    "filter": self._active_filters(),
                }
            },
        }
        try:
            response = await self.es_client.search(index=self.cv_index_name, body=query)
        except NotFoundError:
            logger.info(f"Index {self.cv_index_name} does not exist")
            return []

        return response["hits"]["hits"]

    async def _vector_search(self, query_vector: list, size=5):
        try:
            response = await self.es_client.search(
                index=self.cv_index_name,
                body={
                    "_source": {"excludes": ["embedding_vector"]},
                    "size": size,
                    "query": {
                        "script_score": {
                            "query": {"bool": {"filter": self._active_filters()}},
                            "script": {
                                "source": """
                                    double score = cosineSimilarity(params.query_vector, 'embedding_vector');
                                    if (Double.isNaN(score) || score < -1.0) {
                                        return 0.0;
                                    }
                                    return score + 1.0;
                                """,
                                "params": {"query_vector": query_vector},
                            },
                        }
                    },
                },
            )
        except NotFoundError:
            logger.info(f"Index {self.cv_index_name} does not exist")
            return []

        return response["hits"]["hits"]

    async def _vectors_search(self, query_vector: list, query_text: str = "", size=5):
        candidate_size = max(size * RRF_CANDIDATE_MULTIPLIER, size)
        lexical_hits, vector_hits = await asyncio.gather(
            self._keywords_search(query_text, "", size=candidate_size)
            if query_text
            else asyncio.sleep(0, result=[]),
            self._vector_search(query_vector, size=candidate_size),
        )
        return self._rrf_fuse_hits([lexical_hits, vector_hits], size=size)

    async def match(self, keywords, vector, job_name):
        query_text = f"{job_name} {keywords}".strip()
        hybrid_hits = await self._vectors_search(vector, query_text=query_text)
        if hybrid_hits:
            return hybrid_hits
        return await self._keywords_search(keywords, job_name)

    async def review(
        self, jd_content, jd_keywords, cv_list: list[MatcherData]
    ) -> list[MatcherData]:
        results = []

        for resume in cv_list:
            prompt = PROMPT_REVIEW.format(
                raw_job_description=jd_content,
                extracted_job_keywords=jd_keywords,
                raw_resume=resume.content,
                extracted_resume_keywords=resume.keywords,
            )

            gen_res, _ = await self.model_gen("", prompt, SYSTEM_REVIEW)
            resume.merge_model_result_and_cv_data_original(gen_res)
            results.append(resume)

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
            "id",
        ]

        dict2str = "\n".join(
            [f"{k}: {v}" for k, v in contents.items() if k not in key_rm]
        )
        return dict2str

    async def extract_match_review(
        self, data: bytes | dict, prompt, file_name, jd_id=None
    ):
        if isinstance(data, dict):
            data = await self._pre_data(data)

        if prompt is None:
            prompt = PROMPT

        if file_name:
            suffix = "." + file_name.split(".")[-1]
        else:
            suffix = None

        data_converted = self.preprocess_data.convert_data(data, suffix)
        gen_res, _ = await self.model_gen(data_converted, prompt, SYSTEM)

        # gen_res_format = convert_jd_format(gen_res)
        emb_result = await self.model_embed([data_converted], TASK, query=True)

        cv_top_k_review = None
        ## Match and review
        if gen_res["extracted_keywords"]:
            cv_matcher = await self.match(
                keywords=", ".join(gen_res["extracted_keywords"]),
                vector=emb_result[0],
                job_name=gen_res["job_name"],
            )

            ## Function check if run review or not
            if jd_id is None:
                search_result_past = []
            else:
                search_result_past = await self._get_search_result(jd_id)

            if len(search_result_past) > 0:
                search_result_past = search_result_past[0]["_source"]
            else:
                search_result_past = {"top_cv_id": []}

            cv_matcher_not_reviewed: list[MatcherData] = []
            cv_matcher_reviewed: list[MatcherData] = []
            for v in cv_matcher:
                if v["_source"]["id"] not in search_result_past["top_cv_id"]:
                    cv_matcher_not_reviewed.append(
                        MatcherData.get_cv_data_from_search_by_keyword(v)
                    )

                else:
                    index = search_result_past["top_cv_id"].index(v["_source"]["id"])
                    cv_matcher_reviewed.append(
                        MatcherData.get_cv_info_from_search_in_past(
                            v, search_result_past, index
                        )
                    )

            logger.info(len(cv_matcher_not_reviewed))
            logger.info(len(cv_matcher_reviewed))
            ## Review CV
            cv_top_k_review = await self.review(
                data_converted,
                gen_res["extracted_keywords"],
                cv_matcher_not_reviewed,
            )
            logger.info(f"After run model: {cv_matcher_not_reviewed}")
            ## Merge result in past and current
            cv_top_k_review = cv_top_k_review + cv_matcher_reviewed

            cv_top_k_review = [v for v in cv_top_k_review if v.match_score >= 20]
            cv_top_k_review = sorted(
                cv_top_k_review, key=lambda x: x.match_score, reverse=True
            )

            # Remove cv not have enough confident and create data to save to search-result

            if jd_id:
                logger.info("Saving resume ....")
                try:
                    await self._store_jd(
                        gen_res, emb_result[0], file_name, jd_id, data_converted
                    )
                    await self._store_search_result(jd_id, cv_top_k_review)
                    await self.es_client.close()

                except:
                    logger.info("Save data failed!!!!!!")
                    logger.error(traceback.format_exc())

            cv_top_k_review = [asdict(v) for v in cv_top_k_review]
        else:
            logger.info("Can not get extracted keywords")

        await self.es_client.close()
        return gen_res, cv_top_k_review


_jd_service_instance = None


def get_jd_service() -> JDService:
    global _jd_service_instance
    if _jd_service_instance is None:
        logger.info("Initing JD Service ....")
        _jd_service_instance = JDService()
    return _jd_service_instance


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
