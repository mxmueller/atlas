import uvicorn
import torch.distributed as dist
from fastapi import FastAPI
from routes.analysis import router as analysis_router
from routes.prefilter import router as prefilter_router
from config.settings import settings
from models.llm import LLMSingleton

def create_app():
    app = FastAPI()
    
    LLMSingleton()
    
    app.include_router(analysis_router, tags=["analysis"])
    app.include_router(prefilter_router, tags=["prefilter"])
    
    return app

if __name__ == "__main__":
    try:
        dist.destroy_process_group()
    except:
        pass
    
    app = create_app()
    config = uvicorn.Config(
        app,
        host=settings.HOST,
        port=settings.PORT,
        loop="asyncio",
        workers=1
    )
    server = uvicorn.Server(config)
    server.run()