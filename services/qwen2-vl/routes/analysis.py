from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import asyncio
from vllm import SamplingParams
from tasks.image import process_image  
from tasks.json import parse_json_response 
from models.llm import LLMSingleton

router = APIRouter()

@router.post("/analyze")
async def analyze_ui_element(images: List[UploadFile] = File(...)):
    try:
        image_contents = await asyncio.gather(*[image.read() for image in images])
        batch_inputs = await asyncio.gather(*[process_image(image_content) for image_content in image_contents])

        llm_singleton = LLMSingleton()
        async with llm_singleton._lock:
            outputs = llm_singleton.llm.generate(
                batch_inputs,
                sampling_params=SamplingParams(temperature=0.2, max_tokens=512)
            )

        results = await asyncio.gather(*[parse_json_response(output.outputs[0].text) for output in outputs])
        return JSONResponse(content=results[0] if len(results) == 1 else results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )