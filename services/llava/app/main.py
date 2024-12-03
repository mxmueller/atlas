from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from PIL import Image
import io
import json
import time
import torch
import asyncio
from vllm import SamplingParams 

from app.core.config import Config, logger
from app.models.schemas import UIElement, UIElementMatchInput, UIElementMatch
from app.processors.analyzer import AppState, process_element_batch, clean_json_string


app = FastAPI(title="UI Element Analysis API")
app_state = AppState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    try:
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
        
        status = {
            "status": "healthy",
            "gpu": gpu_info,
            "llm_initialized": app_state.llm is not None,
            "ocr_initialized": app_state.reader is not None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile]) -> List[Dict[str, Any]]:
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    logger.info(f"[{request_id}] Received batch analysis request for {len(files)} files")
    
    if not files:
        logger.error(f"[{request_id}] No files provided")
        raise HTTPException(status_code=400, detail="No files provided")
    
    images = []
    for idx, file in enumerate(files):
        if file.content_type not in Config.ALLOWED_MIME_TYPES:
            logger.warning(f"[{request_id}] Skipping file {idx}: Invalid mime type {file.content_type}")
            continue
            
        try:
            logger.debug(f"[{request_id}] Processing file {idx}: {file.filename}")
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            if image.size[0] > Config.MAX_IMAGE_SIZE[0] or image.size[1] > Config.MAX_IMAGE_SIZE[1]:
                logger.debug(f"[{request_id}] Resizing image {idx} from {image.size} to {Config.MAX_IMAGE_SIZE}")
                image.thumbnail(Config.MAX_IMAGE_SIZE)
            
            images.append(image)
            logger.debug(f"[{request_id}] Successfully processed file {idx}")
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing file {idx}: {str(e)}", exc_info=True)
            continue
    
    if not images:
        logger.error(f"[{request_id}] No valid images found in request")
        raise HTTPException(status_code=400, detail="No valid images provided")
    
    # Process images in batches
    batches = [images[i:i + Config.BATCH_SIZE] 
              for i in range(0, len(images), Config.BATCH_SIZE)]
    
    logger.info(f"[{request_id}] Processing {len(batches)} batches")
    
    results = []
    for batch_idx, batch in enumerate(batches):
        try:
            logger.info(f"[{request_id}] Processing batch {batch_idx + 1}/{len(batches)}")
            batch_results = await process_element_batch(batch, app_state)
            results.extend(batch_results)
            logger.debug(f"[{request_id}] Successfully processed batch {batch_idx + 1}")
        except Exception as e:
            logger.error(f"[{request_id}] Error processing batch {batch_idx + 1}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    duration = time.time() - start_time
    logger.info(f"[{request_id}] Completed batch analysis in {duration:.2f}s")
    return results

@app.post("/match-ui-element", response_model=UIElementMatch)
async def match_ui_element(input_data: UIElementMatchInput) -> UIElementMatch:
    start_time = time.time()
    request_id = f"match_{int(start_time)}"
    logger.info(f"[{request_id}] Received UI element matching request")

    if app_state.llm is None:
        logger.error(f"[{request_id}] LLM not initialized!")
        try:
            app_state.initialize()
            if app_state.llm is None:
                raise ValueError("Failed to initialize LLM model")
        except Exception as e:
            logger.error(f"[{request_id}] LLM initialization failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to initialize LLM model")

    try:
        elements_str = json.dumps(input_data.elements, indent=2)
        
        prompt = f"""USER: Given these UI elements:
{elements_str}

Find the element that best matches this user request: "{input_data.query}"

Think step by step:
1. Analyze the user's request - what are they looking for?
2. For each UI element, evaluate how well it matches what the user wants
3. Consider all properties (visual, semantic, text) but focus on what's most relevant for this specific request
4. If no element is a good match, it's better to return no match than a poor match

Remember:
- A confidence of 0.9+ means an almost perfect match
- A confidence of 0.5-0.8 means a partial match
- A confidence below 0.3 means a poor match
- Use -1 as index if no good match is found

Response format (JSON only):
{{
  "best_match_index": <index of best matching element or -1>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "Clear explanation of why this element matches (or why no good match found)"
}}

ASSISTANT:"""

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=256,
            top_p=0.3,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )

        logger.info(f"[{request_id}] Starting LLM analysis")
        
        llm_output = await asyncio.to_thread(
            lambda: app_state.llm.generate([{"prompt": prompt}], sampling_params=sampling_params)
        )

        raw_response = llm_output[0].outputs[0].text
        logger.info(f"[{request_id}] Raw LLM response: {raw_response}")
        logger.info(f"[{request_id}] LLM output structure: {llm_output}")

        try:
            logger.info(f"[{request_id}] Starting JSON cleaning...")
            cleaned_response = clean_json_string(raw_response)
            logger.info(f"[{request_id}] Cleaned response: {cleaned_response}")
            result = json.loads(cleaned_response)
            logger.info(f"[{request_id}] Parsed result: {result}")
            
            best_match_index = result.get("best_match_index", -1)
            logger.info(f"[{request_id}] Best match index: {best_match_index}")
            
            if best_match_index >= 0 and best_match_index < len(input_data.elements):
                matched_element = input_data.elements[best_match_index]
            else:
                matched_element = {}
            
            response = UIElementMatch(
                element=matched_element,
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", "No matching element found")
            )
            
            duration = time.time() - start_time
            logger.info(f"[{request_id}] Matching completed in {duration:.2f}s")
            logger.debug(f"[{request_id}] Match result: {response}")
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"[{request_id}] Failed to parse LLM response: {str(e)}")
            logger.error(f"[{request_id}] Raw response was: {raw_response}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM response: {str(e)}"
            )

    except Exception as e:
        logger.error(f"[{request_id}] Error during matching: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Matching failed: {str(e)}"
        )
    
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)