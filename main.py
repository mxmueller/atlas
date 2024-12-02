from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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

logging.basicConfig(level=logging.INFO)
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

class AppState:
    def __init__(self):
        self.llm = None
        self.reader = None
        self.thread_pool = ThreadPoolExecutor(max_workers=Config.BATCH_SIZE)
        self.analysis_cache = {}
        self.initialize()

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
                logger.error(f"Error initializing LLM: {e}")
                raise

        if self.reader is None:
            logger.info("Initializing OCR reader")
            try:
                # Initialisiere nur einen Reader für beide Sprachen
                self.reader = easyocr.Reader(['de', 'en'], gpu=True, download_enabled=False)
                logger.info("OCR initialization successful")
            except Exception as e:
                logger.error(f"Error initializing OCR: {e}")
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

async def extract_text_async(image: Image.Image, reader: easyocr.Reader) -> str:
    """Asynchronously extract text using OCR"""
    def _process_ocr(reader: easyocr.Reader, image_np: np.ndarray) -> str:
        try:
            results = reader.readtext(image_np)
            # Vereinfache die Ausgabe zu einem einzigen String
            return " ".join([result[1] for result in results])
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return ""

    image_np = np.array(image)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(app_state.thread_pool, _process_ocr, reader, image_np)
    return text

def extract_visual_features(image: np.ndarray) -> Dict[str, Any]:
    """Extract basic visual features from image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    avg_color = np.mean(image, axis=(0, 1)).astype(int)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return {
        "color": f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}",
        "size": image.shape[:2],
        "aspect_ratio": round(image.shape[1] / image.shape[0], 2),
        "brightness": int(np.mean(gray)),
        "has_border": len(contours) > 0
    }

async def process_element_batch(images: List[Image.Image]) -> List[Dict[str, Any]]:
    """Process a batch of images with both visual and text analysis"""
    if app_state.llm is None or app_state.reader is None:
        logger.error("Models not initialized!")
        app_state.initialize()
        if app_state.llm is None or app_state.reader is None:
            raise HTTPException(status_code=500, detail="Failed to initialize models")

    # Parallel visual feature extraction
    feature_futures = [
        app_state.thread_pool.submit(extract_visual_features, np.array(img))
        for img in images
    ]
    
    # Parallel OCR processing
    ocr_futures = [extract_text_async(img, app_state.reader) for img in images]
    
    # LLM analysis setup
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        top_p=0.95
    )
    
    prompts = [
        {"prompt": get_minimal_analysis_prompt(), "multi_modal_data": {"image": img}}
        for img in images
    ]
    
    try:
        # Run LLM and OCR analysis in parallel
        llm_future = asyncio.to_thread(
            lambda: app_state.llm.generate(prompts, sampling_params=sampling_params)
        )
        
        # Wait for all analysis to complete
        llm_outputs, *ocr_outputs = await asyncio.gather(
            llm_future,
            *ocr_futures
        )
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Combine results
    visual_features = [future.result() for future in feature_futures]
    semantic_features = [output.outputs[0].text for output in llm_outputs]
    
    return [
        {
            "visual": vf,
            "semantic": sf,
            "text": ocr
        }
        for vf, sf, ocr in zip(visual_features, semantic_features, ocr_outputs)
    ]

@app.post("/analyze-batch")
async def analyze_batch(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Analyze batch of UI elements"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    images = []
    for file in files:
        if file.content_type not in Config.ALLOWED_MIME_TYPES:
            continue
            
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(Config.MAX_IMAGE_SIZE)
            images.append(image)
        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            continue
    
    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")
    
    batches = [images[i:i + Config.BATCH_SIZE] 
              for i in range(0, len(images), Config.BATCH_SIZE)]
    
    results = []
    for batch in batches:
        try:
            batch_results = await process_element_batch(batch)
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "llm_initialized": app_state.llm is not None,
            "ocr_initialized": app_state.reader is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)