from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import Dict, List, Any
from PIL import Image
from vllm import LLM, SamplingParams
import easyocr
import numpy as np
import io
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
import json
import time

# Enhanced logging setup
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

def get_minimal_analysis_prompt() -> str:
    return """USER: <image> Analyze this UI element. Provide JSON with type, color, and purpose.
Format: {"type":"","color":"","purpose":""}
ASSISTANT:"""

class UIElement(BaseModel):
    id: str
    visual: Dict[str, Any]
    semantic: Dict[str, Any]
    text: str = ""

class AppState:
    def __init__(self):
        self.llm = None
        self.reader = None
        self.thread_pool = ThreadPoolExecutor(max_workers=Config.BATCH_SIZE)
        self.analysis_cache = {}
        self.initialize()
        logger.info("AppState initialized")

    def initialize(self):
        if self.llm is None:
            logger.info("Initializing LLM model")
            try:
                self.llm = LLM(
                    model=Config.MODEL_NAME,
                    max_model_len=2048,
                    tensor_parallel_size=1,
                    max_num_batched_tokens=4096,
                    max_num_seqs=Config.BATCH_SIZE,
                    gpu_memory_utilization=0.95,
                    enforce_eager=True
                )
                logger.info("LLM initialization successful")
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
                raise

        if self.reader is None:
            logger.info("Initializing OCR reader")
            try:
                self.reader = easyocr.Reader(['de', 'en'], gpu=True, download_enabled=False)
                logger.info("OCR initialization successful")
            except Exception as e:
                logger.error(f"Error initializing OCR: {str(e)}", exc_info=True)
                raise

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
    """Enhanced health check endpoint with detailed status information"""
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
        logger.info(f"Health check: {json.dumps(status, indent=2)}")
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def extract_text_async(image: Image.Image, reader: easyocr.Reader) -> str:
    """Asynchronously extract text using OCR with enhanced logging"""
    logger.info("Starting OCR text extraction")
    
    def _process_ocr(reader: easyocr.Reader, image_np: np.ndarray) -> str:
        try:
            start_time = time.time()
            results = reader.readtext(image_np)
            duration = time.time() - start_time
            
            text = " ".join([result[1] for result in results])
            logger.info(f"OCR completed in {duration:.2f}s")
            logger.debug(f"OCR results: {results}")
            logger.debug(f"Extracted text: '{text}'")
            return text
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}", exc_info=True)
            return ""

    try:
        image_np = np.array(image)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(app_state.thread_pool, _process_ocr, reader, image_np)
        return text
    except Exception as e:
        logger.error(f"Error in extract_text_async: {str(e)}", exc_info=True)
        return ""

def extract_visual_features(image: np.ndarray) -> Dict[str, Any]:
    """Extract visual features from image with detailed logging"""
    logger.info("Starting visual feature extraction")
    start_time = time.time()
    
    try:
        # Convert to HSV and grayscale
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate average color
        avg_color = np.mean(image, axis=(0, 1)).astype(int)
        color_hex = f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}"
        
        # Edge detection and contour analysis
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate features
        features = {
            "color": color_hex,
            "size": image.shape[:2],
            "aspect_ratio": round(image.shape[1] / image.shape[0], 2),
            "brightness": int(np.mean(gray)),
            "has_border": len(contours) > 0,
            "edge_density": len(np.where(edges > 0)[0]) / (image.shape[0] * image.shape[1]),
            "contour_count": len(contours)
        }
        
        duration = time.time() - start_time
        logger.info(f"Visual feature extraction completed in {duration:.2f}s")
        logger.debug(f"Extracted features: {json.dumps(features, indent=2)}")
        return features
        
    except Exception as e:
        logger.error(f"Error in visual feature extraction: {str(e)}", exc_info=True)
        return {
            "color": "#000000",
            "size": (0, 0),
            "aspect_ratio": 1.0,
            "brightness": 0,
            "has_border": False,
            "edge_density": 0.0,
            "contour_count": 0,
            "error": str(e)
        }


async def process_element_batch(images: List[Image.Image], user_prompt: str = "") -> List[Dict[str, Any]]:
    batch_start = time.time()
    batch_id = f"batch_{int(batch_start)}"
    logger.info(f"[{batch_id}] Starting batch processing of {len(images)} images with prompt: {user_prompt}")

    feature_futures = [
        app_state.thread_pool.submit(extract_visual_features, np.array(img))
        for img in images
    ]
    
    ocr_futures = [extract_text_async(img, app_state.reader) for img in images]
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=256,
        top_p=0.95
    )
    
    prompts = [
        {"prompt": Config.get_prompt(user_prompt), "multi_modal_data": {"image": img}}
        for img in images
    ]
    
    try:
        logger.info(f"[{batch_id}] Starting LLM analysis")
        llm_start = time.time()
        
        llm_future = asyncio.to_thread(
            lambda: app_state.llm.generate(prompts, sampling_params=sampling_params)
        )
        
        llm_outputs, *ocr_outputs = await asyncio.gather(
            llm_future,
            *ocr_futures
        )
        
        llm_duration = time.time() - llm_start
        logger.info(f"[{batch_id}] LLM analysis completed in {llm_duration:.2f}s")
        
    except Exception as e:
        logger.error(f"[{batch_id}] Error during parallel processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    try:
        visual_features = [future.result() for future in feature_futures]
        semantic_features = [
            json.loads(clean_json_string(output.outputs[0].text)) if output.outputs[0].text.strip() else {}
            for output in llm_outputs
        ]

        results = [
            {
                "visual": vf,
                "semantic": sf,
                "text": ocr,
                "match_score": sf.get("matches_user_intent", {}).get("score", 0.0) if sf else 0.0
            }
            for vf, sf, ocr in zip(visual_features, semantic_features, ocr_outputs)
        ]
        
        batch_duration = time.time() - batch_start
        logger.info(f"[{batch_id}] Batch processing completed in {batch_duration:.2f}s")
        logger.debug(f"[{batch_id}] Batch results: {json.dumps(results, indent=2)}")
        
        return results
        
    except Exception as e:
        logger.error(f"[{batch_id}] Error combining results: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to combine results: {str(e)}")
@app.post("/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    user_prompt: str = Query(default="", description="User prompt for image analysis")
) -> List[Dict[str, Any]]:
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    logger.info(f"[{request_id}] Received batch analysis request for {len(files)} files with prompt: {user_prompt}")
    
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
                image.thumbnail(Config.MAX_IMAGE_SIZE)
            
            images.append(image)
            logger.debug(f"[{request_id}] Successfully processed file {idx}")
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing file {idx}: {str(e)}", exc_info=True)
            continue
    
    if not images:
        logger.error(f"[{request_id}] No valid images found in request")
        raise HTTPException(status_code=400, detail="No valid images provided")
    
    batches = [images[i:i + Config.BATCH_SIZE] 
              for i in range(0, len(images), Config.BATCH_SIZE)]
    
    logger.info(f"[{request_id}] Processing {len(batches)} batches")
    
    results = []
    for batch_idx, batch in enumerate(batches):
        try:
            logger.info(f"[{request_id}] Processing batch {batch_idx + 1}/{len(batches)}")
            batch_results = await process_element_batch(batch, user_prompt=user_prompt)
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
    """Find the best matching UI element based on a natural language query"""
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
        # Format UI elements for prompt
        elements_str = json.dumps(input_data.elements, indent=2)
        
        # Create improved structured prompt
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

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,    # Low but not zero to allow some flexibility
            max_tokens=256,     
            top_p=0.3,         # More focused sampling
            presence_penalty=0.0,
            frequency_penalty=0.0
        )

        logger.info(f"[{request_id}] Starting LLM analysis")
        
        # Perform LLM analysis
        llm_output = await asyncio.to_thread(
            lambda: app_state.llm.generate([{"prompt": prompt}], sampling_params=sampling_params)
        )

        # Get and process response
        raw_response = llm_output[0].outputs[0].text
        logger.info(f"[{request_id}] Raw LLM response: {raw_response}")  # Changed to info for better visibility
        logger.info(f"[{request_id}] LLM output structure: {llm_output}")  # Log complete structure

        try:
            logger.info(f"[{request_id}] Starting JSON cleaning...")
            cleaned_response = clean_json_string(raw_response)
            logger.info(f"[{request_id}] Cleaned response: {cleaned_response}")
            logger.info(f"[{request_id}] Type of cleaned response: {type(cleaned_response)}")
            logger.info(f"[{request_id}] Attempting to parse JSON...")
            result = json.loads(cleaned_response)
            logger.info(f"[{request_id}] Parsed result: {result}")
            logger.info(f"[{request_id}] Result type: {type(result)}")
            
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