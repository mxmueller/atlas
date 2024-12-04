from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from vllm import LLM, SamplingParams
import uvicorn
import logging
import torch.distributed as dist
import json
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
from functools import partial

logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class LLMSingleton:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMSingleton, cls).__new__(cls)
            cls._instance.llm = LLM(
                model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
                trust_remote_code=True,
                dtype="float16",
                max_model_len=32768,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                max_num_batched_tokens=32768,
                max_num_seqs=64
            )
            cls._instance.executor = ThreadPoolExecutor(max_workers=4)
        return cls._instance

app = FastAPI()
llm_singleton = LLMSingleton()

def create_analysis_prompt() -> str:
    base_prompt = (
        "<|im_start|>system\n"
        "You are a precise UI element analyzer. Extract information ONLY from what you can see.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Example output format:\n"
        "{\n"
        '    "type": "button|icon|text|input",\n'
        '    "text": "exact text if present, null if none",\n'
        '    "visual_elements": ["icon names or descriptions if present else put none"],\n'
        '    "primary_function": "main purpose based on visual evidence only make two sentence",\n'
        '    "dominant_color": "main color if clearly visible, null if unclear"\n'
        "}\n\n"
    )
    return (
        f"{base_prompt}"
        "<|vision_start|>"
        "<|image_pad|>"
        "<|vision_end|>\n"
        "Analyze this UI element and return valid JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

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
        logging.error(f"Error processing image: {e}")
        raise

async def parse_json_response(response_text: str) -> dict:
    """Parse JSON response asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            llm_singleton.executor,
            json.loads,
            response_text
        )
    except json.JSONDecodeError as e:
        return {"error": "Invalid JSON response", "raw_response": response_text}

@app.post("/analyze")
async def analyze_ui_element(
    images: List[UploadFile] = File(...)
):
    try:
        # Read all images asynchronously
        image_contents = await asyncio.gather(
            *[image.read() for image in images]
        )

        # Process images in parallel
        batch_inputs = await asyncio.gather(
            *[process_image(image_content) for image_content in image_contents]
        )

        # Perform batch inference
        async with llm_singleton._lock:
            outputs = llm_singleton.llm.generate(
                batch_inputs,
                sampling_params=SamplingParams(
                    temperature=0.2,
                    max_tokens=512
                )
            )

        # Parse results asynchronously
        results = await asyncio.gather(
            *[parse_json_response(output.outputs[0].text) for output in outputs]
        )

        # Return single result if only one image was provided
        if len(results) == 1:
            return JSONResponse(content=results[0])

        return JSONResponse(content=results)

    except Exception as e:
        logging.error(f"Error in batch processing: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

if __name__ == "__main__":
    try:
        dist.destroy_process_group()
    except:
        pass
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        workers=1  # vLLM handles its own parallelization
    )
    server = uvicorn.Server(config)
    server.run()