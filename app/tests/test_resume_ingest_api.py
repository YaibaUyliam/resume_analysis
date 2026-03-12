import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.resume import resume_extract_router
from app.agent import get_resume_service


class FakeResumeService:
    async def ingest_resume(self, contents, file_name, cv_id=None, sys_mess=None):
        return {
            "file_name": file_name,
            "cv_id": cv_id,
            "stored_cv_id": cv_id or "generated-id",
            "info_extract": {"personalInfo": {"fullName": "Test User"}},
            "info_extract_raw": {"personal_info": {"full_name": "Test User"}},
            "service_resp": {"eval_count": 10},
            "cv_data_converted": "resume body",
            "emb_result": [0.1, 0.2],
            "is_duplicate": False,
            "duplicate_status": "new_candidate",
            "similarity_score": 0.0,
            "existing_cv_id": None,
            "existing_cv_url": None,
            "duplicate_candidates": [],
            "version_action": "store_new",
            "stored": True,
            "reasons": ["no_duplicate_match"],
        }


class ResumeIngestApiTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(resume_extract_router, prefix="/api/resumes")
        self.app.dependency_overrides[get_resume_service] = lambda: FakeResumeService()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_ingest_endpoint_returns_unified_response(self):
        response = self.client.post(
            "/api/resumes/ingest",
            files={"cv_file": ("resume.txt", b"hello world", "text/plain")},
            data={"cv_id": "cv-123"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["stored_cv_id"], "cv-123")
        self.assertEqual(body["duplicate_status"], "new_candidate")
        self.assertIn("info_extract", body)
        self.assertIn("emb_result", body)

    def test_ingest_rejects_large_file(self):
        large_payload = b"a" * (11 * 1024 * 1024)

        response = self.client.post(
            "/api/resumes/ingest",
            files={"cv_file": ("resume.txt", large_payload, "text/plain")},
        )

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
