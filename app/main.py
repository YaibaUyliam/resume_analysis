import os
import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

from app.core import setup_logging
from app.api import health_check, router_func

from app.agent import ResumeService, get_resume_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    resume_service = get_resume_service()
    yield
    await resume_service.close()


def create_app() -> FastAPI:
    """
    Create the FastAPI application instance.
    """

    setup_logging()

    app = FastAPI(lifespan=lifespan)

    app.include_router(health_check)
    app.include_router(router_func)

    global_lock = asyncio.Lock()
    if int(os.environ.get("REQUEST_LIMIT", 0)) == 1:

        @app.middleware("http")
        async def single_request_middleware(request, call_next):
            async with global_lock:
                return await call_next(request)

    return app


app = create_app()


# if __name__ == "__main__":
#     uvicorn.run()
