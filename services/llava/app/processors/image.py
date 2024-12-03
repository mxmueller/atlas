from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
import easyocr
import asyncio
from PIL import Image
import numpy as np
import json
import time
from typing import Dict, Any
import cv2
from ..core.config import Config, logger

async def extract_text_async(image: Image.Image, reader: easyocr.Reader) -> str:
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
        text = await loop.run_in_executor(None, _process_ocr, reader, image_np)
        return text
    except Exception as e:
        logger.error(f"Error in extract_text_async: {str(e)}", exc_info=True)
        return ""

def extract_visual_features(image: np.ndarray) -> Dict[str, Any]:
    logger.info("Starting visual feature extraction")
    start_time = time.time()
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg_color = np.mean(image, axis=(0, 1)).astype(int)
        color_hex = f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}"
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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