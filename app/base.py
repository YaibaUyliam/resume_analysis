from contextlib import asynccontextmanager

from fastapi import FastAPI

from .agent import GenerationManager
from .core import setup_logging
from .api import health_check, router_func


# create lifespan, start db
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load all model
    # if use ollama, only create client, server already started
    app.state.model_gen = GenerationManager()
    yield


def create_app() -> FastAPI:
    """
    Create the FastAPI application instance.
    """

    setup_logging()

    app = FastAPI(
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )

    app.include_router(health_check)
    app.include_router(router_func)

    return app