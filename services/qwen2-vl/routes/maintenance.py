from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.llm import LLMSingleton

router = APIRouter()

@router.post("/reset")
async def reset_llm():
    try:
        llm = LLMSingleton()
        memory_stats = await llm.reset()
        return JSONResponse(content={
            "status": "success",
            "memory_stats": memory_stats
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )