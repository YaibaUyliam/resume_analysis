import unittest

from app.agent.utils import (
    build_resume_identity,
    build_structured_duplicate_score,
    compare_resume_versions,
)
from app.agent.resume_service import ResumeService
from app.agent.jd_service import JDService


def make_resume_info(
    *,
    full_name="Nguyen Van A",
    email="nguyenvana@example.com",
    phone="+84 912 345 678",
    location="Ho Chi Minh City",
    desired_position="Backend Engineer",
    year_of_experience="5 years",
    school="HCMUT",
    degree="Bachelor",
    company="ACME",
    position="Software Engineer",
    skills=None,
    keywords=None,
):
    return {
        "personal_info": {
            "full_name": full_name,
            "email": email,
            "phone_number": phone,
            "current_location": location,
            "desired_position": desired_position,
            "year_of_experience": year_of_experience,
            "linkedin_url": "https://linkedin.com/in/nguyenvana",
            "github_url": "https://github.com/nguyenvana",
            "year_of_birth": "1998",
        },
        "education": [
            {
                "school_name": school,
                "major": "Computer Science",
                "degree": degree,
                "duration": "2016-2020",
            }
        ],
        "experience": [
            {
                "company": company,
                "position": position,
                "duration": "2020-Now",
                "job_description": "Built backend APIs",
            }
        ],
        "skills": [
            {"skill_name": skill, "proficiency": "", "summary_skill": ""}
            for skill in (skills or ["Python", "FastAPI", "Elasticsearch"])
        ],
        "project": [],
        "certificates": [],
        "extracted_keywords": keywords or ["Python", "FastAPI", "Elasticsearch"],
    }


class ResumeUtilsTest(unittest.TestCase):
    def test_build_resume_identity_normalizes_core_fields(self):
        info = make_resume_info()
        identity = build_resume_identity(info, "resume body")

        self.assertEqual(identity["email_normalized"], "nguyenvana@example.com")
        self.assertEqual(identity["phone_normalized"], "0912345678")
        self.assertEqual(identity["linkedin_normalized"], "nguyenvana")
        self.assertEqual(identity["github_normalized"], "nguyenvana")
        self.assertIn("Backend Engineer", identity["canonical_cv_text"])
        self.assertTrue(identity["content_hash"])

    def test_structured_duplicate_score_is_high_for_same_candidate(self):
        incoming = build_resume_identity(make_resume_info(), "resume body")
        existing = build_resume_identity(
            make_resume_info(
                full_name="Nguyễn Văn A",
                phone="0912345678",
                company="ACME Corp",
                position="Backend Engineer",
                skills=["Python", "FastAPI", "Docker"],
            ),
            "resume body updated",
        )

        score = build_structured_duplicate_score(
            incoming, existing, vector_similarity=0.91
        )
        self.assertGreaterEqual(score["score"], 0.85)
        self.assertGreater(score["breakdown"]["name"], 0.9)

    def test_compare_resume_versions_detects_same_version_by_hash(self):
        incoming = build_resume_identity(make_resume_info(), "same body")
        existing = build_resume_identity(make_resume_info(), "same body")

        version_result = compare_resume_versions(incoming, existing)
        self.assertTrue(version_result["same_version"])
        self.assertEqual(version_result["reason"], "content_hash_match")

    def test_compare_resume_versions_detects_meaningful_change(self):
        incoming = build_resume_identity(make_resume_info(company="NewCo"), "new body")
        existing = build_resume_identity(make_resume_info(company="OldCo"), "old body")

        version_result = compare_resume_versions(incoming, existing)
        self.assertFalse(version_result["same_version"])

    def test_resume_rrf_fusion_prioritizes_documents_seen_by_both_rankers(self):
        service = ResumeService.__new__(ResumeService)
        lexical_hits = [
            {"_id": "a", "_score": 5.0, "_source": {"id": "a"}},
            {"_id": "b", "_score": 4.0, "_source": {"id": "b"}},
        ]
        vector_hits = [
            {"_id": "b", "_score": 0.9, "_source": {"id": "b"}},
            {"_id": "c", "_score": 0.8, "_source": {"id": "c"}},
        ]

        fused_hits = service._rrf_fuse_hits([lexical_hits, vector_hits], size=3)

        self.assertEqual(fused_hits[0]["_source"]["id"], "b")
        self.assertGreater(fused_hits[0]["rrf_score"], fused_hits[1]["rrf_score"])

    def test_jd_rrf_fusion_prioritizes_documents_seen_by_both_rankers(self):
        service = JDService.__new__(JDService)
        lexical_hits = [
            {"_id": "cv-1", "_score": 3.0, "_source": {"id": "cv-1"}},
            {"_id": "cv-2", "_score": 2.0, "_source": {"id": "cv-2"}},
        ]
        vector_hits = [
            {"_id": "cv-2", "_score": 0.9, "_source": {"id": "cv-2"}},
            {"_id": "cv-3", "_score": 0.8, "_source": {"id": "cv-3"}},
        ]

        fused_hits = service._rrf_fuse_hits([lexical_hits, vector_hits], size=3)

        self.assertEqual(fused_hits[0]["_source"]["id"], "cv-2")
        self.assertGreater(fused_hits[0]["rrf_score"], fused_hits[1]["rrf_score"])


if __name__ == "__main__":
    unittest.main()
