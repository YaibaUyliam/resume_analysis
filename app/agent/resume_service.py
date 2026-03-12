from __future__ import annotations

import asyncio
import math
import traceback
import re
import os
import uuid

from loguru import logger
from datetime import datetime, timezone, timedelta
from elasticsearch import AsyncElasticsearch, ConflictError, NotFoundError
from elastic_transport import ConnectionError as ElasticConnectionError
from pydantic import BaseModel
from typing import Any, List, Optional

from app.agent.utils import (
    build_canonical_cv_text,
    build_resume_identity,
    build_structured_duplicate_score,
    compare_resume_versions,
    convert_resume_format,
)
from app.agent.providers import (
    PreprocessData,
    OllamaExtractionProvider,
    OllamaEmbeddingProvider,
)
from app.agent.providers.prompt.resume_prompt import PROMPT, SYSTEM, TASK
from app.core.setting import settings

RRF_K = 60
RRF_CANDIDATE_MULTIPLIER = 3

SYSTEM_DUPLICATE = """You compare two resume records and respond ONLY with valid JSON.
Schema:
{
  "same_candidate": true,
  "same_version": false,
  "new_version": true,
  "confidence": 0.0,
  "reasons": []
}
Rules:
- same_candidate=true only if both resumes are very likely the same person.
- same_version=true only if the new resume is effectively the same version as the stored one.
- new_version=true only if same_candidate=true and there are meaningful resume updates.
- confidence is a float from 0 to 1.
- reasons is an array of concise strings.
"""


class ResumeSchema(BaseModel):
    id: str
    cv_url: str
    content: str
    keywords: Optional[str]
    year_of_experience: Optional[float]
    embedding_vector: List[float]
    full_name: Optional[str]
    desired_position: Optional[str]
    created_at: str
    updated_at: str
    is_deleted: bool = False
    is_active: bool = True
    version_group_id: Optional[str] = None
    version_number: int = 1
    superseded_by: Optional[str] = None
    duplicate_strategy_version: str = "v1"
    canonical_cv_text: str = ""
    content_hash: str = ""
    full_name_normalized: str = ""
    email_normalized: str = ""
    phone_normalized: str = ""
    linkedin_normalized: str = ""
    github_normalized: str = ""
    location_normalized: str = ""
    current_location: Optional[str] = None
    year_of_birth: Optional[int] = None
    latest_company_normalized: str = ""
    latest_position_normalized: str = ""
    latest_school_normalized: str = ""
    latest_degree_normalized: str = ""
    skills: list[str] = []
    extracted_keywords: list[str] = []


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
        self.similar_thresh = settings.SIMILAR_THRESH
        self.auto_merge_thresh = settings.DUPLICATE_AUTO_MERGE_THRESHOLD
        self.llm_thresh = settings.DUPLICATE_LLM_THRESHOLD
        self.duplicate_candidate_size = settings.DUPLICATE_CANDIDATE_SIZE

        self.es_client = AsyncElasticsearch(hosts=[settings.ES_HOST])
        self.index_name = settings.ES_CV_INDEX
        self._index_ready = False
        logger.info(f"Index name: {self.index_name}")

        self.timezone = timezone(timedelta(hours=8))

    def _now_iso(self) -> str:
        return datetime.now(self.timezone).isoformat()

    def _resume_index_mapping(self, embedding_dims: int) -> dict:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "cv_url": {"type": "keyword"},
                    "content": {"type": "text"},
                    "keywords": {"type": "text"},
                    "year_of_experience": {"type": "float"},
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": False,
                    },
                    "full_name": {"type": "text"},
                    "desired_position": {"type": "text"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "is_deleted": {"type": "boolean"},
                    "is_active": {"type": "boolean"},
                    "version_group_id": {"type": "keyword"},
                    "version_number": {"type": "integer"},
                    "superseded_by": {"type": "keyword"},
                    "duplicate_strategy_version": {"type": "keyword"},
                    "canonical_cv_text": {"type": "text"},
                    "content_hash": {"type": "keyword"},
                    "full_name_normalized": {"type": "keyword"},
                    "email_normalized": {"type": "keyword"},
                    "phone_normalized": {"type": "keyword"},
                    "linkedin_normalized": {"type": "keyword"},
                    "github_normalized": {"type": "keyword"},
                    "location_normalized": {"type": "keyword"},
                    "current_location": {"type": "text"},
                    "year_of_birth": {"type": "integer"},
                    "latest_company_normalized": {"type": "keyword"},
                    "latest_position_normalized": {"type": "keyword"},
                    "latest_school_normalized": {"type": "keyword"},
                    "latest_degree_normalized": {"type": "keyword"},
                    "skills": {"type": "keyword"},
                    "extracted_keywords": {"type": "keyword"},
                }
            },
        }

    async def _ensure_resume_index(self, embedding_dims: int) -> None:
        if self._index_ready:
            return

        exists = await self.es_client.indices.exists(index=self.index_name)
        if exists:
            self._index_ready = True
            return

        try:
            await self.es_client.indices.create(
                index=self.index_name,
                body=self._resume_index_mapping(embedding_dims),
            )
            logger.info(
                f"Created Elasticsearch index {self.index_name} with dims={embedding_dims}"
            )
        except ConflictError:
            logger.info(f"Elasticsearch index {self.index_name} already exists")

        self._index_ready = True

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

    def _get_file_suffix(self, file_name: str) -> str:
        return "." + file_name.split(".")[-1].lower()

    async def _prepare_resume_data(self, data, file_name) -> tuple[str | list[str], str]:
        suffix = self._get_file_suffix(file_name)
        data_converted = self.preprocess_data.convert_data(data, suffix)
        return data_converted, suffix

    async def _build_embedding(
        self, resume_input: str, query: bool = False
    ) -> list[float]:
        emb_result = await self.model_embed([resume_input], TASK, query=query)
        return emb_result[0]

    async def _extract_resume(
        self, data_converted: str | list[str], sys_mess: Optional[str]
    ):
        if sys_mess is None:
            sys_mess = SYSTEM
        gen_res, service_resp = await self.model_extract(data_converted, PROMPT, sys_mess)
        gen_res_format = convert_resume_format(gen_res)
        return gen_res, gen_res_format, service_resp

    def _extract_identity_from_source(self, source: dict) -> dict:
        return {
            "full_name": source.get("full_name", ""),
            "full_name_normalized": source.get("full_name_normalized", ""),
            "email_normalized": source.get("email_normalized", ""),
            "phone_normalized": source.get("phone_normalized", ""),
            "linkedin_normalized": source.get("linkedin_normalized", ""),
            "github_normalized": source.get("github_normalized", ""),
            "year_of_birth": source.get("year_of_birth"),
            "current_location": source.get("current_location", ""),
            "location_normalized": source.get("location_normalized", ""),
            "desired_position": source.get("desired_position", ""),
            "years_of_experience": source.get("year_of_experience"),
            "latest_company_normalized": source.get("latest_company_normalized", ""),
            "latest_position_normalized": source.get("latest_position_normalized", ""),
            "latest_school_normalized": source.get("latest_school_normalized", ""),
            "latest_degree_normalized": source.get("latest_degree_normalized", ""),
            "skills": source.get("skills", []),
            "keywords": source.get("extracted_keywords", []),
            "canonical_cv_text": source.get("canonical_cv_text", ""),
            "content_hash": source.get("content_hash", ""),
        }

    def _build_term_queries(self, identity: dict) -> list[dict]:
        queries = []
        term_fields = [
            "email_normalized",
            "phone_normalized",
            "linkedin_normalized",
            "github_normalized",
        ]
        for field in term_fields:
            value = identity.get(field)
            if value:
                queries.append({"term": {field: value}})
        return queries

    def _build_text_queries(self, identity: dict, query_text: str = "") -> list[dict]:
        queries = self._build_term_queries(identity)
        if query_text:
            queries.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "keywords^2",
                            "desired_position^2",
                            "full_name^1.5",
                            "canonical_cv_text",
                            "content",
                        ],
                    }
                }
            )
        return queries

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
                current_source = current["hit"].get("_source", {})
                hit_source = hit.get("_source", {})
                if (
                    isinstance(hit_source.get("embedding_vector"), list)
                    and not isinstance(current_source.get("embedding_vector"), list)
                ) or hit.get("_score", 0.0) > current["hit"].get("_score", 0.0):
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

    async def _lexical_search(
        self, identity: dict, query_text: str = "", size: int = 20
    ) -> list[dict]:
        query_should = self._build_text_queries(identity, query_text)
        if not query_should:
            return []

        try:
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "_source": True,
                    "size": size,
                    "query": {
                        "bool": {
                            "filter": self._active_filters(),
                            "should": query_should,
                            "minimum_should_match": 1,
                        }
                    },
                },
            )
        except NotFoundError:
            logger.info(f"Index {self.index_name} does not exist")
            return []
        return response["hits"]["hits"]

    async def _vector_search(self, query_vector: list[float], size: int = 20) -> list[dict]:
        try:
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "_source": True,
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
            logger.info(f"Index {self.index_name} does not exist")
            return []
        return response["hits"]["hits"]

    async def _find_exact_identity_matches(self, identity: dict, size: int = 5):
        should_queries = self._build_term_queries(identity)
        if not should_queries:
            return []

        try:
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "_source": {"excludes": ["embedding_vector"]},
                    "size": size,
                    "query": {
                        "bool": {
                            "filter": self._active_filters(),
                            "should": should_queries,
                            "minimum_should_match": 1,
                        }
                    },
                },
            )
        except NotFoundError:
            logger.info(
                f"Elasticsearch index {self.index_name} does not exist yet; skipping exact duplicate lookup"
            )
            return []
        return response["hits"]["hits"]

    async def _hybrid_search(
        self,
        query_vector: list[float],
        identity: Optional[dict] = None,
        query_text: str = "",
        size: int = 20,
    ):
        identity = identity or {}
        await self._ensure_resume_index(len(query_vector))
        candidate_size = max(size * RRF_CANDIDATE_MULTIPLIER, size)
        lexical_hits, vector_hits = await asyncio.gather(
            self._lexical_search(identity, query_text=query_text, size=candidate_size),
            self._vector_search(query_vector, size=candidate_size),
        )
        return self._rrf_fuse_hits([lexical_hits, vector_hits], size=size)

    def _cosine_similarity(self, vector_a: list[float], vector_b: list[float]) -> float:
        if not vector_a or not vector_b or len(vector_a) != len(vector_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _vector_similarity_from_hit(
        self, hit: dict, query_vector: Optional[list[float]] = None
    ) -> float:
        source = hit.get("_source", {})
        embedding_vector = source.get("embedding_vector", [])
        if query_vector and isinstance(embedding_vector, list):
            return self._cosine_similarity(query_vector, embedding_vector)
        return 0.0

    async def _llm_duplicate_judgement(
        self,
        incoming_identity: dict,
        existing_source: dict,
        structured_score: dict,
        version_result: dict,
    ) -> dict:
        existing_identity = self._extract_identity_from_source(existing_source)
        prompt = (
            "New resume identity:\n"
            f"{incoming_identity}\n\n"
            "Existing resume identity:\n"
            f"{existing_identity}\n\n"
            "Structured score:\n"
            f"{structured_score}\n\n"
            "Version comparison:\n"
            f"{version_result}\n"
        )
        judgement, _ = await self.model_extract("", prompt, SYSTEM_DUPLICATE)
        if not isinstance(judgement, dict):
            return {
                "same_candidate": False,
                "same_version": False,
                "new_version": False,
                "confidence": 0.0,
                "reasons": ["invalid_llm_output"],
            }
        return judgement

    async def _deactivate_existing_resume(self, hit: dict, new_cv_id: str):
        await self.es_client.update(
            index=self.index_name,
            id=hit["_id"],
            doc={
                "is_active": False,
                "superseded_by": new_cv_id,
                "updated_at": self._now_iso(),
            },
        )

    def _normalize_year_of_experience(self, gen_res: dict) -> float | None:
        year_of_experience = gen_res["personal_info"]["year_of_experience"]
        if year_of_experience:
            match = re.search(r"\d+", str(year_of_experience))
            if match:
                return float(match.group(0))
        return None

    def _build_resume_document(
        self,
        gen_res: dict,
        emb_res: list[float],
        file_name: str,
        cv_id: str,
        resume_data: str | list[str],
        version_group_id: Optional[str] = None,
        version_number: int = 1,
        superseded_by: Optional[str] = None,
        is_active: bool = True,
    ) -> ResumeSchema:
        identity = build_resume_identity(gen_res, resume_data)
        time_current = self._now_iso()
        doc = {
            "id": cv_id,
            "cv_url": file_name,
            "content": resume_data if isinstance(resume_data, str) else "",
            "keywords": ", ".join(gen_res["extracted_keywords"]),
            "year_of_experience": self._normalize_year_of_experience(gen_res),
            "embedding_vector": emb_res,
            "full_name": gen_res["personal_info"]["full_name"],
            "desired_position": gen_res["personal_info"].get("desired_position"),
            "created_at": time_current,
            "updated_at": time_current,
            "is_active": is_active,
            "version_group_id": version_group_id or cv_id,
            "version_number": version_number,
            "superseded_by": superseded_by,
            "canonical_cv_text": identity["canonical_cv_text"],
            "content_hash": identity["content_hash"],
            "full_name_normalized": identity["full_name_normalized"],
            "email_normalized": identity["email_normalized"],
            "phone_normalized": identity["phone_normalized"],
            "linkedin_normalized": identity["linkedin_normalized"],
            "github_normalized": identity["github_normalized"],
            "location_normalized": identity["location_normalized"],
            "current_location": identity["current_location"],
            "year_of_birth": identity["year_of_birth"],
            "latest_company_normalized": identity["latest_company_normalized"],
            "latest_position_normalized": identity["latest_position_normalized"],
            "latest_school_normalized": identity["latest_school_normalized"],
            "latest_degree_normalized": identity["latest_degree_normalized"],
            "skills": identity["skills"],
            "extracted_keywords": gen_res["extracted_keywords"],
        }
        return ResumeSchema(**doc)

    async def _store_resume(self, gen_res, emb_res, file_name, cv_id, resume_data, **kwargs):
        await self._ensure_resume_index(len(emb_res))
        resume_extract_result = self._build_resume_document(
            gen_res, emb_res, file_name, cv_id, resume_data, **kwargs
        )
        resp = await self.es_client.index(
            index=self.index_name, document=resume_extract_result.model_dump()
        )
        logger.info(resp)
        return resp, resume_extract_result

    async def del_cv(self, cv_id):
        try:
            search_resp = await self.es_client.search(
                index=self.index_name,
                query={"term": {"id": cv_id}},
                source=["_id"],
            )
        except NotFoundError:
            logger.info(f"Index {self.index_name} does not exist")
            return None

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
        return await self._hybrid_search(query_vector, query_text="", size=size)

    async def _detect_duplicate_four_layers(
        self,
        incoming_identity: dict,
        emb_result: list[float],
        generated_resume: dict,
    ) -> dict:
        exact_matches = await self._find_exact_identity_matches(incoming_identity)
        for hit in exact_matches:
            existing_source = hit["_source"]
            version_result = compare_resume_versions(
                incoming_identity, self._extract_identity_from_source(existing_source)
            )
            if version_result["same_version"]:
                return {
                    "is_duplicate": True,
                    "duplicate_status": "same_candidate_same_version",
                    "similarity_score": 1.0,
                    "existing_cv_id": existing_source["id"],
                    "existing_cv_url": existing_source["cv_url"],
                    "existing_hit": hit,
                    "version_action": "keep_old",
                    "should_store": False,
                    "duplicate_candidates": [
                        {
                            "cv_id": existing_source["id"],
                            "cv_url": existing_source["cv_url"],
                            "similarity_score": 1.0,
                            "decision_layer": "layer1_exact",
                        }
                    ],
                    "reasons": ["exact_identity_match", version_result["reason"]],
                }
            return {
                "is_duplicate": True,
                "duplicate_status": "same_candidate_new_version",
                "similarity_score": 1.0,
                "existing_cv_id": existing_source["id"],
                "existing_cv_url": existing_source["cv_url"],
                "existing_hit": hit,
                "version_action": "replace_old",
                "should_store": True,
                "duplicate_candidates": [
                    {
                        "cv_id": existing_source["id"],
                        "cv_url": existing_source["cv_url"],
                        "similarity_score": 1.0,
                        "decision_layer": "layer1_exact",
                    }
                ],
                "reasons": ["exact_identity_match", version_result["reason"]],
            }

        query_text = " ".join(
            [
                build_canonical_cv_text(generated_resume),
                " ".join(generated_resume.get("extracted_keywords", [])),
            ]
        ).strip()
        candidate_hits = await self._hybrid_search(
            emb_result,
            identity=incoming_identity,
            query_text=query_text,
            size=self.duplicate_candidate_size,
        )

        duplicate_candidates = []
        gray_zone = []
        for hit in candidate_hits:
            existing_source = hit["_source"]
            existing_identity = self._extract_identity_from_source(existing_source)
            vector_similarity = self._vector_similarity_from_hit(hit, emb_result)
            structured_score = build_structured_duplicate_score(
                incoming_identity, existing_identity, vector_similarity
            )
            version_result = compare_resume_versions(incoming_identity, existing_identity)
            duplicate_candidates.append(
                {
                    "cv_id": existing_source["id"],
                    "cv_url": existing_source["cv_url"],
                    "similarity_score": structured_score["score"],
                    "vector_similarity": structured_score["vector_similarity"],
                    "existing_cv_id": existing_source["id"],
                    "decision_layer": "layer2_structured",
                }
            )
            if structured_score["score"] >= self.auto_merge_thresh:
                return {
                    "is_duplicate": True,
                    "duplicate_status": (
                        "same_candidate_same_version"
                        if version_result["same_version"]
                        else "same_candidate_new_version"
                    ),
                    "similarity_score": structured_score["score"],
                    "existing_cv_id": existing_source["id"],
                    "existing_cv_url": existing_source["cv_url"],
                    "existing_hit": hit,
                    "version_action": (
                        "keep_old" if version_result["same_version"] else "replace_old"
                    ),
                    "should_store": not version_result["same_version"],
                    "duplicate_candidates": duplicate_candidates,
                    "reasons": ["structured_score_threshold"],
                }

            if structured_score["score"] >= self.llm_thresh:
                gray_zone.append((hit, structured_score, version_result))

        for hit, structured_score, version_result in gray_zone[:3]:
            existing_source = hit["_source"]
            judgement = await self._llm_duplicate_judgement(
                incoming_identity, existing_source, structured_score, version_result
            )
            if judgement.get("same_candidate"):
                is_same_version = bool(
                    judgement.get("same_version") or version_result["same_version"]
                )
                return {
                    "is_duplicate": True,
                    "duplicate_status": (
                        "same_candidate_same_version"
                        if is_same_version
                        else "same_candidate_new_version"
                    ),
                    "similarity_score": structured_score["score"],
                    "existing_cv_id": existing_source["id"],
                    "existing_cv_url": existing_source["cv_url"],
                    "existing_hit": hit,
                    "version_action": "keep_old" if is_same_version else "replace_old",
                    "should_store": not is_same_version,
                    "duplicate_candidates": duplicate_candidates,
                    "reasons": judgement.get("reasons", []),
                    "llm_confidence": judgement.get("confidence", 0.0),
                }

        return {
            "is_duplicate": False,
            "duplicate_status": "new_candidate",
            "similarity_score": 0.0,
            "existing_cv_id": None,
            "existing_cv_url": None,
            "existing_hit": None,
            "version_action": "store_new",
            "should_store": True,
            "duplicate_candidates": duplicate_candidates,
            "reasons": ["no_duplicate_match"],
        }

    def _build_duplicate_unavailable_result(self, exc: Exception) -> dict[str, Any]:
        error_message = str(exc) or exc.__class__.__name__
        return {
            "is_duplicate": False,
            "duplicate_status": "duplicate_check_unavailable",
            "similarity_score": 0.0,
            "existing_cv_id": None,
            "existing_cv_url": None,
            "existing_hit": None,
            "version_action": "skip_store",
            "should_store": False,
            "duplicate_candidates": [],
            "reasons": [f"elasticsearch_unavailable: {error_message}"],
        }

    async def ingest_resume(
        self,
        data,
        file_name: str,
        cv_id: Optional[str] = None,
        sys_mess: Optional[str] = None,
    ) -> dict[str, Any]:
        stored_cv_id = cv_id or str(uuid.uuid4())
        data_converted, _ = await self._prepare_resume_data(data, file_name)
        gen_res, gen_res_format, service_resp = await self._extract_resume(
            data_converted, sys_mess
        )
        canonical_resume = build_canonical_cv_text(gen_res)
        embed_input = canonical_resume or (
            data_converted if isinstance(data_converted, str) else "\n".join(data_converted)
        )
        emb_result = await self._build_embedding(embed_input)
        incoming_identity = build_resume_identity(gen_res, data_converted)
        try:
            duplicate_result = await self._detect_duplicate_four_layers(
                incoming_identity, emb_result, gen_res
            )
        except ElasticConnectionError as exc:
            logger.warning(
                f"Skipping duplicate detection because Elasticsearch is unavailable: {exc}"
            )
            duplicate_result = self._build_duplicate_unavailable_result(exc)

        stored_resp = None
        try:
            if duplicate_result["should_store"]:
                version_group_id = stored_cv_id
                version_number = 1
                existing_hit = duplicate_result.get("existing_hit")
                if existing_hit is not None:
                    existing_source = existing_hit["_source"]
                    version_group_id = (
                        existing_source.get("version_group_id") or existing_source["id"]
                    )
                    version_number = int(existing_source.get("version_number", 1)) + 1
                    if duplicate_result["version_action"] == "replace_old":
                        await self._deactivate_existing_resume(existing_hit, stored_cv_id)

                stored_resp, _ = await self._store_resume(
                    gen_res,
                    emb_result,
                    file_name,
                    stored_cv_id,
                    data_converted,
                    version_group_id=version_group_id,
                    version_number=version_number,
                )
            else:
                stored_cv_id = duplicate_result.get("existing_cv_id") or stored_cv_id
        except ElasticConnectionError as exc:
            logger.warning(
                f"Skipping resume storage because Elasticsearch is unavailable: {exc}"
            )
            duplicate_result = self._build_duplicate_unavailable_result(exc)

        return {
            "file_name": file_name,
            "cv_id": cv_id,
            "stored_cv_id": stored_cv_id,
            "info_extract": gen_res_format,
            "info_extract_raw": gen_res,
            "service_resp": service_resp.model_dump(),
            "cv_data_converted": data_converted,
            "emb_result": emb_result[:10],
            "is_duplicate": duplicate_result["is_duplicate"],
            "duplicate_status": duplicate_result["duplicate_status"],
            "similarity_score": duplicate_result["similarity_score"],
            "existing_cv_id": duplicate_result["existing_cv_id"],
            "existing_cv_url": duplicate_result["existing_cv_url"],
            "duplicate_candidates": duplicate_result["duplicate_candidates"],
            "version_action": duplicate_result["version_action"],
            "stored": stored_resp is not None,
            "reasons": duplicate_result.get("reasons", []),
        }

    async def check_duplication(
        self, data, file_name
    ) -> tuple[list, str | list[str], list[float]]:
        try:
            data_converted, _ = await self._prepare_resume_data(data, file_name)
            embed_input = (
                data_converted if isinstance(data_converted, str) else "\n".join(data_converted)
            )
            emb_result = await self._build_embedding(embed_input)
            embed_query_result = await self._hybrid_search(
                emb_result,
                query_text=embed_input[:1000],
                size=2,
            )

            check_result = []
            for cv_info in embed_query_result:
                cosine_sim = self._vector_similarity_from_hit(cv_info, emb_result)
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
        data_converted, _ = await self._prepare_resume_data(data, file_name)
        gen_res, gen_res_format, service_resp = await self._extract_resume(
            data_converted, sys_mess
        )
        return gen_res, gen_res_format, service_resp

    # Already convert data in step check duplication
    async def extract_and_store(self, data, file_name, cv_id, cv_embed):
        try:
            gen_res, gen_res_format, _ = await self._extract_resume(data, SYSTEM)

            logger.info("Saving resume ....")
            await self._store_resume(gen_res, cv_embed, file_name, cv_id, data)

        except Exception:
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


def get_existing_resume_service() -> Optional[ResumeService]:
    return _resume_service_instance
