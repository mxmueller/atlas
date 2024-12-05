import json
import asyncio
from models.llm import LLMSingleton

async def parse_json_response(response_text: str) -> dict:
    """Parse JSON response asynchronously."""
    try:
        llm_singleton = LLMSingleton()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            llm_singleton.executor,
            json.loads,
            response_text
        )
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON response", "raw_response": response_text}
