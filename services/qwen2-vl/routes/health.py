from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.llm import LLMSingleton

router = APIRouter()

@router.get("/health")
async def health_check():
   try:
       llm = LLMSingleton()
       stats = llm._get_detailed_memory_stats()
       return JSONResponse(status_code=200, content={"status": "ok", "stats": stats})
   except Exception as e:
       return JSONResponse(
           status_code=503,
           content={"status": "unavailable", "error": str(e)}
       )