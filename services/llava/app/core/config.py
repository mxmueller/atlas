import logging
from typing import Dict, Any, Set, Tuple
import torch
import json

def parse_llm_response(text: str) -> Dict[str, Any]:
    try:
        # Find JSON content
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return {}
            
        json_str = text[start:end+1]
        
        # Remove escaped underscores
        json_str = json_str.replace('\\_', '_')
        
        # Basic cleanup
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        json_str = ' '.join(json_str.split())
        
        # Parse JSON
        data = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {data}")
        return data
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}\nText was: {text}")
        return {}

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    BATCH_SIZE = 2
    MAX_IMAGE_SIZE = (400, 400)
    ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_prompt(user_prompt: str = "") -> str:
        base_prompt = """USER: <image> Analyze this UI element and compare it with the user's request.

User is looking for: {user_prompt}

Provide detailed JSON about its visual and functional characteristics, and evaluate if it matches the user's request.

Format:
{{
    "type": "",
    "color": "",
    "purpose": "",
    "description": "Brief 1 sentence description explaining the element's function and appearance",
    "matches_user_intent": {{
        "score": 0.0,
        "reasoning": "Explain why this element does or doesn't match the user's request"
    }}
}}

Focus on key UI/UX aspects and evaluate how well this element matches what the user is looking for.
Only respond with valid JSON, no escaped characters or formatting.
ASSISTANT:"""
        return base_prompt.format(user_prompt=user_prompt if user_prompt else "any UI element")