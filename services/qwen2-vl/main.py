import uvicorn
import torch.distributed as dist
from fastapi import FastAPI
from routes import analysis, prefilter, match
from config.settings import settings
from models.llm import LLMSingleton

def create_app():
    app = FastAPI()
    
    LLMSingleton()

    app.include_router(analysis.router, prefix="/api/v1")
    app.include_router(prefilter.router, prefix="/api/v1")
    app.include_router(match.router, prefix="/api/v1")
    
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