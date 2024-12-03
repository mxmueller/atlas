from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams  # Added SamplingParams import
import easyocr
import asyncio
from PIL import Image
import numpy as np
import json
import time
from typing import List, Dict, Any
from ..core.config import Config, logger
from .image import extract_visual_features, extract_text_async

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

def clean_json_string(s: str) -> str:
    s = s.strip().strip('"')
    s = s.replace('\\_', '_')
    s = s.replace('\_', '_')
    s = s.replace('\\"', '"')
    
    if '{' in s:
        s = s[s.find('{'):s.rfind('}')+1]
    
    logger.debug(f"Cleaned JSON string: {s}")
    return s

async def process_element_batch(images: List[Image.Image], app_state: AppState) -> List[Dict[str, Any]]:
    batch_start = time.time()
    batch_id = f"batch_{int(batch_start)}"
    logger.info(f"[{batch_id}] Starting batch processing of {len(images)} images")

    feature_futures = [
        app_state.thread_pool.submit(extract_visual_features, np.array(img))
        for img in images
    ]
    
    ocr_futures = [extract_text_async(img, app_state.reader) for img in images]
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        top_p=0.95
    )
    
    prompts = [
        {"prompt": Config.get_prompt(), "multi_modal_data": {"image": img}}
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
        raise

    try:
        visual_features = [future.result() for future in feature_futures]
        semantic_features = [
            json.loads(output.outputs[0].text) if output.outputs[0].text.strip() else {}
            for output in llm_outputs
        ]

        results = [
            {
                "visual": vf,
                "semantic": sf,  
                "text": ocr
            }
            for vf, sf, ocr in zip(visual_features, semantic_features, ocr_outputs)
        ]
        
        batch_duration = time.time() - batch_start
        logger.info(f"[{batch_id}] Batch processing completed in {batch_duration:.2f}s")
        logger.debug(f"[{batch_id}] Batch results: {json.dumps(results, indent=2)}")
        
        return results
        
    except Exception as e:
        logger.error(f"[{batch_id}] Error combining results: {str(e)}", exc_info=True)
        raise