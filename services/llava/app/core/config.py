import logging
from typing import Set, Tuple
import torch

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
    def get_prompt() -> str:
        return """USER: <image> Analyze this UI element. Provide detailed JSON about its visual and functional characteristics.
            Format:
            {
                "type": "",
                "color": "",
                "purpose": "",
                "description": "Brief 1-2 sentence description explaining the element's function and appearance.",
                "interaction": "How users would interact with this element",
                "context": "Where this element likely appears and its role"
            }

            Keep descriptions concise but informative. Focus on key UI/UX aspects.
            ASSISTANT:"""