import os
import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

from .core import setup_logging
from .api import health_check, router_func


# create lifespan, start db
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # load all model
#     # if use ollama, only create client, server already started
#     generation_manager = GenerationManager()
#     app.state.model_gen = await generation_manager.init_model()

#     embedding_manager = EmbeddingManager()
#     app.state.model_emb = await embedding_manager.init_model()

#     yield


def create_app() -> FastAPI:
    """
    Create the FastAPI application instance.
    """

    setup_logging()

    app = FastAPI()  # , lifespan=lifespan)

    app.include_router(health_check)
    app.include_router(router_func)

    global_lock = asyncio.Lock()
    if int(os.environ.get("REQUEST_LIMIT", 0)) == 1:

        @app.middleware("http")
        async def single_request_middleware(request, call_next):
            async with global_lock:
                return await call_next(request)

    return app
