from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import hashlib
from io import BytesIO
from datetime import datetime
import base64
from typing import Dict, List

app = FastAPI()
mongo_client = AsyncIOMotorClient("mongodb://mongo:27017")
db = mongo_client.cache_db
cache_collection = db.image_cache

MASK_API_URL = "http://mask-generation:8000/api/artifacts"
QWEN_API_NORMALIZE_URL = "http://qwen2-vl:8000/api/v1/normalize"
QWEN_API_FILTER_URL = "http://qwen2-vl:8000/api/v1/prefilter"
QWEN_API_ANALYZE_URL = "http://qwen2-vl:8000/api/v1/analyze"

def clean_nested_children(section):
    result = section.copy()
    if "children" in result:
        cleaned_children = []
        for child in result["children"]:
            cleaned_child = {k: v for k, v in child.items() 
                           if k not in ["has_children", "children_count", "children"]}
            cleaned_child.pop("score", None)
            cleaned_child.pop("label", None)
            cleaned_children.append(cleaned_child)
        result["children"] = cleaned_children
    return result

def prepare_section_for_json(section: Dict) -> Dict:
    result = section.copy()
    if result.get("has_children") and "children" in result:
        cleaned_children = []
        for child in result["children"]:
            child_copy = child.copy()
            if "neighbors" in child_copy:
                for direction, neighbor in child_copy["neighbors"].items():
                    if neighbor is not None and isinstance(neighbor, dict):
                        child_copy["neighbors"][direction] = {
                            "id": neighbor["id"],
                            "box": neighbor["box"],
                            "type": neighbor.get("type"),
                            "text": neighbor.get("text"),
                            "visual_elements": neighbor.get("visual_elements"),
                            "dominant_color": neighbor.get("dominant_color")
                        }
            cleaned_children.append(child_copy)
        result["children"] = cleaned_children
    return result

def prepare_mask_result_for_json(mask_result: Dict) -> Dict:
    result = mask_result.copy()
    result["sections"] = [prepare_section_for_json(section) for section in result["sections"]]
    return result

async def get_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()

def build_section_map(sections: List[Dict]) -> Dict:
    section_map = {}
    def add_section_recursive(section: Dict):
        section_map[section["id"]] = section
        if section.get("has_children") and section.get("children"):
            for child in section["children"]:
                add_section_recursive(child)
    for section in sections:
        add_section_recursive(section)
    return section_map

async def analyze_sections(sections: List[Dict]) -> List[Dict]:
    if not sections:
        return []
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        for i, section in enumerate(sections):
            if "image" in section:
                image_bytes = base64.b64decode(section["image"])
                data.add_field('images', image_bytes, filename=f'image{i}.jpg', content_type='image/jpeg')

        async with session.post(QWEN_API_ANALYZE_URL, data=data) as response:
            if response.status != 200:
                error_body = await response.text()
                raise HTTPException(500, f"Analysis failed: {error_body}")
                
            results = await response.json()
            if not isinstance(results, list):
                results = [results]
                
            for section, result in zip(sections, results):
                section.update({
                    "type": result.get("type"),
                    "text": result.get("text"), 
                    "visual_elements": result.get("visual_elements"),
                    "primary_function": result.get("primary_function"),
                    "dominant_color": result.get("dominant_color")
                })
                section.pop("score", None)
                section.pop("label", None)
                
                if section.get("children"):
                    for child in section["children"]:
                        if isinstance(child, dict):
                            child.pop("score", None)
                            child.pop("label", None)
                            child.pop("has_children", None)
                            child.pop("children_count", None)
                            child.pop("children", None)
            
            return sections

def resolve_children_neighbors(mask_result: Dict, section_map: Dict):
    for section in mask_result["sections"]:
        if section.get("has_children") and "children" in section:
            for child in section["children"]:
                if "neighbors" in child:
                    for direction in ["left", "right", "above", "below"]:
                        if child["neighbors"][direction]:
                            neighbor_id = child["neighbors"][direction]
                            if neighbor_id in section_map:
                                child["neighbors"][direction] = section_map[neighbor_id]

async def process_sections(mask_result: Dict, normalized_prompt: str):
    section_map = build_section_map(mask_result["sections"])
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

    resolve_children_neighbors(mask_result, section_map)

    sections_to_analyze = []
    for section in filtered_sections:
        sections_to_analyze.append(section)
        if section.get("has_children") and section.get("children"):
            for child in section["children"]:
                sections_to_analyze.append(child)

    analyzed_sections = await analyze_sections(sections_to_analyze)
    for section in analyzed_sections:
        section_map[section["id"]] = section

    return {
        "filtered_section_ids": [section["id"] for section in filtered_sections],
        "analyzed_section_ids": [section["id"] for section in analyzed_sections]
    }

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), prompt: str = Form(...),
    include_mask: bool = Query(False), analyzed_data: bool = Query(False)):
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    content = await file.read()
    image_hash = await get_image_hash(content)
    
    cached_result = await cache_collection.find_one({"image_hash": image_hash})
    if cached_result:
        mask_result = cached_result["result"]
    else:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', BytesIO(content), filename=file.filename, content_type=file.content_type)
            
            async with session.post(MASK_API_URL, data=form) as response:
                if response.status != 200:
                    raise HTTPException(500, "Mask generation failed")
                mask_result = await response.json()
                
            await cache_collection.insert_one({
                "image_hash": image_hash,
                "result": mask_result,
                "created_at": datetime.utcnow()
            })

    async with aiohttp.ClientSession() as session:
        async with session.post(QWEN_API_NORMALIZE_URL, json={"prompt": prompt}) as response:
            if response.status != 200:
                raise HTTPException(500, "Prompt normalization failed")
            normalized_prompt = await response.json()

    process_result = await process_sections(mask_result, normalized_prompt)
    filtered_ids = process_result["filtered_section_ids"]
    analyzed_ids = process_result["analyzed_section_ids"]
    
    response = {"filtered_section_ids": filtered_ids}
    
    if analyzed_data:
        analyzed_sections = []
        for section in mask_result["sections"]:
            if section["id"] in analyzed_ids:
                analyzed_sections.append(prepare_section_for_json(section))
        response["analyzed_data"] = analyzed_sections
        
    if include_mask:
        response["mask_result"] = prepare_mask_result_for_json(mask_result)
        
    return response