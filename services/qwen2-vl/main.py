import uvicorn
import torch.distributed as dist
from fastapi import FastAPI
from routes.analysis import router
from config.settings import settings
from models.llm import LLMSingleton

def create_app():
    app = FastAPI()
    
    # Initialize LLM model at startup
    LLMSingleton()
    
    app.include_router(router)
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