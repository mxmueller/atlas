from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import hashlib
from io import BytesIO
import logging
from datetime import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = AsyncIOMotorClient("mongodb://mongo:27017")
db = mongo_client.cache_db
cache_collection = db.image_cache

MASK_API_URL = "http://mask-generation:8000/api/artifacts"
QWEN_API_NORMALIZE_URL = "http://qwen2-vl:8000/api/v1/normalize"
QWEN_API_FILTER_URL = "http://qwen2-vl:8000/api/v1/prefilter"
QWEN_API_MATCH_URL = "http://qwen2-vl:8000/api/v1/match"

async def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()

def resolve_neighbors(element, all_elements):
    neighbors = {
        "left": None,
        "right": None,
        "above": None,
        "below": None
    }
    
    if 'neighbors' in element:
        for direction in ['left', 'right', 'above', 'below']:
            neighbor_id = element['neighbors'].get(direction)
            if neighbor_id:
                neighbor = next((e for e in all_elements if e['id'] == neighbor_id), None)
                if neighbor:
                    neighbors[direction] = {
                        'id': neighbor['id'],
                        'type': 'button',
                        'text': neighbor.get('label', ''),
                        'visual_elements': [],
                        'primary_function': neighbor.get('label', ''),
                        'dominant_color': '',
                        'neighbours': {}
                    }
    return neighbors

async def process_sections(mask_result, normalized_prompt):
    filtered_sections = []
    
    async with aiohttp.ClientSession() as session:
        for section in mask_result["sections"]:
            data = {
                "normalized_prompt": normalized_prompt,
                "sections": [{
                    "position_metadata": section["position_metadata"],
                    "image": section["image"]
                }]
            }
            
            async with session.post(QWEN_API_FILTER_URL, json=data) as response:
                if response.status != 200:
                    continue
                result = await response.json()
                if result["results"][0]["likely_contains"]:
                    filtered_sections.append(section)

    elements_for_matching = []
    for section in filtered_sections:
        if not section.get('children'):
            continue
            
        for child in section['children']:
            if child.get('children_count', 0) > 0:
                continue
                
            element = {
                'id': child['id'],
                'type': 'button',
                'text': child.get('label', ''),
                'visual_elements': [],
                'primary_function': child.get('label', ''),
                'dominant_color': 'red',  # Default
                'neighbours': resolve_neighbors(child, section['children']),
                'image': child.get('image')
            }
            print("DEBUG - Element structure:", element)
            
            element['neighbours'] = resolve_neighbors(child, section['children'])
            elements_for_matching.append(element)

    if not elements_for_matching:
        return None

    async with aiohttp.ClientSession() as session:
        match_data = {
            "normalized_prompt": {
                "type": normalized_prompt.get("type", "button"),
                "text": normalized_prompt.get("text", ""),
                "color": normalized_prompt.get("color", ""),
                "position": normalized_prompt.get("position", "")
            },
            "elements": elements_for_matching
        }
        print("DEBUG - Full match request:", match_data)
        
        async with session.post(QWEN_API_MATCH_URL, json=match_data) as response:
            if response.status != 200:
                return None
                
            match_result = await response.json()
            matched_element = next(
                (e for e in elements_for_matching if e['id'] == match_result['match_id']),
                None
            )
            
            if matched_element:
                return {
                    'match_id': match_result['match_id'],
                    'confidence_score': match_result['confidence_score'],
                    'image': matched_element['image']
                }

    return None

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), prompt: str = Form(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    content = await file.read()
    image_hash = await get_image_hash(content)
    
    cached_result = await cache_collection.find_one({"image_hash": image_hash})
    if cached_result:
        logger.info(f"Cache HIT for hash: {image_hash}")
        mask_result = cached_result["result"]
    else:
        logger.info(f"Cache MISS for hash: {image_hash}")
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', 
                        BytesIO(content), 
                        filename=file.filename,
                        content_type=file.content_type)
            
            async with session.post(MASK_API_URL, data=form) as response:
                if response.status != 200:
                    error_content = await response.text()
                    logger.error(f"Mask generation failed with status {response.status}: {error_content}")
                    raise HTTPException(500, "Mask generation failed")
                mask_result = await response.json()
                
            await cache_collection.insert_one({
                "image_hash": image_hash,
                "result": mask_result,
                "created_at": datetime.utcnow()
            })

    async with aiohttp.ClientSession() as session:
        logger.info(f"Normalizing prompt: {prompt}")
        async with session.post(QWEN_API_NORMALIZE_URL, json={"prompt": prompt}) as response:
            if response.status != 200:
                error_content = await response.text()
                logger.error(f"Prompt normalization failed with status {response.status}: {error_content}")
                raise HTTPException(500, "Prompt normalization failed")
            normalized_prompt = await response.json()

    final_result = await process_sections(mask_result, normalized_prompt)
    if not final_result:
        raise HTTPException(404, "No matching element found")
        
    return final_result