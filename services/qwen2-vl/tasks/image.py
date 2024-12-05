from PIL import Image
import io
import contextlib
from tasks.prompt import create_analysis_prompt

async def process_image(image_data: bytes) -> dict:
    """Process a single image asynchronously."""
    try:
        with contextlib.closing(Image.open(io.BytesIO(image_data))) as img:
            img = img.convert("RGB")
            return {
                "prompt": create_analysis_prompt(),
                "multi_modal_data": {"image": img}
            }
    except Exception as e:
        raise
