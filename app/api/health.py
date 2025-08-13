from fastapi import APIRouter, status, Depends


health_check = APIRouter()


@health_check.get("/healthcheck", tags=["Health check"], status_code=status.HTTP_200_OK)
async def check():
    """
    health check endpoint
    """
    res = "Fine"

    return {"message": "pong", "result": res}
