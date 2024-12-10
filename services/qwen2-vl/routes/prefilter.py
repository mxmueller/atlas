from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import asyncio
from vllm import SamplingParams
from tasks.image import process_image
from tasks.json import parse_json_response
from models.llm import LLMSingleton
from tasks.prompt import create_prefilter_prompt


router = APIRouter()

class PositionMetadata(BaseModel):
    y_start: float
    y_end: float
    vertical_position: str

class Section(BaseModel):
    position_metadata: PositionMetadata
    image: str

class PrefilterRequest(BaseModel):
    normalized_prompt: Dict[str, Any]
    sections: List[Section]

@router.post("/prefilter")
async def prefilter_sections(request: PrefilterRequest):
    try:
        sections = request.sections
        batch_inputs = []
        
        for section in sections:
            image_data = base64.b64decode(section.image)
            processed = await process_image(image_data)
            processed["prompt"] = create_prefilter_prompt(request.normalized_prompt)
            batch_inputs.append(processed)

        llm_singleton = LLMSingleton()
        async with llm_singleton._lock:
            outputs = llm_singleton.llm.generate(
                batch_inputs,
                sampling_params=SamplingParams(temperature=0.1, max_tokens=128)
            )

        results = []
        for idx, output in enumerate(outputs):
            parsed = await parse_json_response(output.outputs[0].text)
            results.append({
                "section_index": idx,
                "position_metadata": sections[idx].position_metadata.dict(),
                "likely_contains": parsed.get("contains", False)
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))