from fastapi import APIRouter

from app.api.resume import resume_extract_router
# from app.api.jd import jd_matcher_router


router_func = APIRouter(prefix="/api", tags=["api"])
router_func.include_router(resume_extract_router, prefix="/resumes")
# router_func.include_router(jd_matcher_router, prefix="/jd")
