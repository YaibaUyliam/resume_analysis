from fastapi import APIRouter

from .resume import resume_extract_router
from .jd import jd_matcher_router


router_func = APIRouter(prefix="/api", tags=["api"])
router_func.include_router(resume_extract_router, prefix="/resumes")
router_func.include_router(jd_matcher_router, prefix="/jd")
