import traceback
import re
import os

from loguru import logger
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from elasticsearch import Elasticsearch, AsyncElasticsearch

from .manager import GenerationManager, EmbeddingManager
from .utils import convert_jd_format
from .providers.prompt.jd_prompt import PROMPT, SYSTEM, TASK
from .providers.prompt.resume_review import PROMPT_REVIEW, SYSTEM_REVIEW


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
        self.generation_manager = GenerationManager()
        self.embedding_manager = EmbeddingManager()

        self.es_client = AsyncElasticsearch(hosts=[os.environ["ES_HOST"]])
        self.jd_index_name = os.environ["ES_JD_INDEX"]
        self.search_result_index_name = os.environ["ES_SEARCH_RESULT_INDEX"]
        self.cv_index_name = os.environ["ES_CV_INDEX"]
        logger.info(f"Index name: {self.jd_index_name, self.cv_index_name}")

        self.timezone = timezone(timedelta(hours=8))

    async def _store_jd(self, gen_res, emb_res, file_name, jd_id, jd_text):
        minimum_years_of_experience = gen_res.get("minimum_years_of_experience", "")
        if minimum_years_of_experience:
            match = re.search(r"\d+", str(minimum_years_of_experience))
            if match:
                minimum_years_of_experience = float(match.group(0))

        doc = {
            "id": jd_id,
            "jd_url": file_name,
            "content": jd_text,
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
        response = await self.es_client.search(
            index=self.search_result_index_name, body=query
        )

        return response["hits"]["hits"]

    async def _keywords_search(self, keywords, job_name, size=5):
        # query = {
        #     "_source": {"excludes": ["embedding_vector"]},
        #     "size": size,
        #     "query": {"match": {"keywords": keywords}},
        # }
        query_content = job_name + keywords
        logger.info(f"Query content: {query_content}")

        query = {
            "_source": {"excludes": ["embedding_vector"]},
            "size": size,
            "query": {
                "multi_match": {
                    "query": query_content,
                    "fields": ["keywords", "desired_position^2"],
                }
            },
        }
        response = await self.es_client.search(index=self.cv_index_name, body=query)

        return response["hits"]["hits"]

    async def _vectors_search(self, query_vector: list, size=5):
        response = await self.es_client.search(
            index=self.cv_index_name,
            body={
                "_source": {"excludes": ["embedding_vector"]},
                "size": size,
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

    async def match(self, keywords, vector, job_name):
        keywords_search_res = await self._keywords_search(keywords, job_name)
        # vectors_search_res = await self._vectors_search(vector)

        # cv_list = []
        # for hit in vectors_search_res:
        #     cv_list.append(hit)

        return keywords_search_res

    async def review(
        self, model_gen, jd_content, jd_keywords, cv_list: list[MatcherData]
    ) -> list[MatcherData]:
        results = []

        for resume in cv_list:
            prompt = PROMPT_REVIEW.format(
                raw_job_description=jd_content,
                extracted_job_keywords=jd_keywords,
                raw_resume=resume.content,
                extracted_resume_keywords=resume.keywords,
            )

            gen_res, _ = await model_gen("", prompt, SYSTEM_REVIEW, None)
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
        self, contents: bytes | dict, prompt, file_name, jd_id=None
    ):
        if isinstance(contents, dict):
            contents = await self._pre_data(contents)

        model_gen = await self.generation_manager.init_model()
        model_emb = await self.embedding_manager.init_model()

        if prompt is None:
            prompt = PROMPT

        if file_name:
            suffix = "." + file_name.split(".")[-1]
        else:
            suffix = None
        gen_res, jd_text = await model_gen(contents, prompt, SYSTEM, suffix)

        # gen_res_format = convert_jd_format(gen_res)
        emb_res = await model_emb([jd_text], TASK, query=True)

        cv_top_k_review = None
        ## Match and review
        if gen_res["extracted_keywords"]:
            cv_matcher = await self.match(
                keywords=", ".join(gen_res["extracted_keywords"]),
                vector=emb_res[0],
                job_name=gen_res["job_name"],
            )

            ## Function check if run review or not
            search_result_past = await self._get_search_result(jd_id)
            logger.info(search_result_past)
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
                model_gen,
                jd_text,
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
                    await self._store_jd(gen_res, emb_res[0], file_name, jd_id, jd_text)
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
