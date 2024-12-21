from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import hashlib
from io import BytesIO
from datetime import datetime
import base64
from typing import Dict, List
import asyncio

app = FastAPI()
mongo_client = AsyncIOMotorClient("mongodb://mongo:27017")
db = mongo_client.cache_db
cache_collection = db.image_cache

MASK_API_URL = "http://mask-generation:8000/api/artifacts"
QWEN_API_NORMALIZE_URL = "http://qwen2-vl:8000/api/v1/normalize" 
QWEN_API_FILTER_URL = "http://qwen2-vl:8000/api/v1/prefilter"
QWEN_API_ANALYZE_URL = "http://qwen2-vl:8000/api/v1/analyze"
QWEN_API_MATCH_URL = "http://qwen2-vl:8000/api/v1/match"

def prepare_element_for_match(element: Dict) -> Dict:
    clean_element = {
        "id": element["id"],
        "type": element.get("type", ""),
        "visual_elements": element.get("visual_elements", []),
        "text": element.get("text"),
        "dominant_color": element.get("dominant_color", "unknown")
    }
    if element.get("primary_function"):
        clean_element["primary_function"] = element["primary_function"]
    if "neighbors" in element:
        clean_neighbors = {}
        for direction, neighbor in element["neighbors"].items():
            if neighbor and isinstance(neighbor, dict):
                neighbor_data = {
                    "id": neighbor["id"],
                    "type": neighbor.get("type", ""),
                    "visual_elements": neighbor.get("visual_elements", []),
                    "dominant_color": neighbor.get("dominant_color", "unknown"),
                    "text": neighbor.get("text", "")
                }
                if neighbor_data["id"]:
                    clean_neighbors[direction] = neighbor_data
        if clean_neighbors:
            clean_element["neighbors"] = clean_neighbors
    return clean_element

async def do_initial_matches(children: List[Dict], normalized_prompt: str) -> List[Dict]:
    cleaned_children = [prepare_element_for_match(child) for child in children]
    batches = []
    for i in range(0, len(cleaned_children), 5):
        batch = cleaned_children[i:i + 5]
        batches.append(batch)
    
    matches = []
    async with aiohttp.ClientSession() as session:
        for batch in batches:
            data = {
                "normalized_prompt": normalized_prompt,
                "elements": batch
            }
            try:
                async with session.post(QWEN_API_MATCH_URL, json=data) as response:
                    if response.status != 200:
                        continue
                    result = await response.json()
                    if result.get("match_id"):
                        matched_element = next(
                            (elem for elem in children if elem["id"] == result["match_id"]),
                            None
                        )
                        if matched_element:
                            matches.append(matched_element)
            except Exception:
                continue
    print(f"DEBUG - Initial matches found: {len(matches)}")
    return matches

async def reduce_matches(matches: List[Dict], normalized_prompt: str) -> Dict:
    current_matches = matches
    while len(current_matches) > 1:
        batches = []
        for i in range(0, len(current_matches), 5):
            batch = current_matches[i:i + 5]
            batches.append(batch)
        
        new_matches = []
        async with aiohttp.ClientSession() as session:
            for batch in batches:
                cleaned_batch = [prepare_element_for_match(match) for match in batch]
                data = {
                    "normalized_prompt": normalized_prompt,
                    "elements": cleaned_batch
                }
                try:
                    async with session.post(QWEN_API_MATCH_URL, json=data) as response:
                        if response.status != 200:
                            continue
                        result = await response.json()
                        if result.get("match_id"):
                            matched_element = next(
                                (elem for elem in batch if elem["id"] == result["match_id"]),
                                None
                            )
                            if matched_element:
                                new_matches.append(matched_element)
                except Exception:
                    continue
        
        if not new_matches:
            break
            
        print(f"DEBUG - Reduced to {len(new_matches)} matches")  # <-- HIER das print einfügen
        current_matches = new_matches
    
    return current_matches[0] if current_matches else None

async def collect_children_for_matching(filtered_sections: List[Dict]):
    all_children = []
    for section in filtered_sections:
        if section.get("has_children") and section.get("children"):
            for child in section["children"]:
                all_children.append(child)
    return all_children

def prepare_neighbor_data(neighbor: Dict) -> Dict:
    neighbor_data = {
        "id": neighbor["id"],
        "box": neighbor["box"],
        "type": neighbor.get("type", ""),
        "visual_elements": neighbor.get("visual_elements", []),
        "dominant_color": neighbor.get("dominant_color", "unknown"),
        "text": neighbor.get("text", ""),
        "neighbors": {}
    }
    
    if "neighbors" in neighbor:
        for sub_direction, sub_neighbor_id in neighbor["neighbors"].items():
            if isinstance(sub_neighbor_id, str):
                neighbor_data["neighbors"][sub_direction] = {
                    "id": sub_neighbor_id
                }
            elif isinstance(sub_neighbor_id, dict):
                neighbor_data["neighbors"][sub_direction] = {
                    "id": sub_neighbor_id.get("id", "")
                }
            
    return neighbor_data

def resolve_children_neighbors(mask_result: Dict, section_map: Dict):
    for section in mask_result["sections"]:
        if section.get("has_children") and "children" in section:
            for child in section["children"]:
                if "neighbors" in child:
                    for direction in ["left", "right", "above", "below"]:
                        if child["neighbors"].get(direction):
                            neighbor_id = child["neighbors"][direction]
                            if neighbor_id in section_map:
                                child["neighbors"][direction] = prepare_neighbor_data(section_map[neighbor_id])

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
    
    analyzed_sections = []
    batch_size = 20
    total_batches = len(sections) // batch_size + (1 if len(sections) % batch_size else 0)
    
    print(f"Starting analysis of {len(sections)} sections in {total_batches} batches")
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(sections), batch_size):
            current_batch = sections[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            print(f"Processing batch {batch_number}/{total_batches} with {len(current_batch)} sections")
            
            try:
                data = aiohttp.FormData()
                image_count = 0
                total_image_size = 0
                
                for j, section in enumerate(current_batch):
                    if "image" in section:
                        image_bytes = base64.b64decode(section["image"])
                        total_image_size += len(image_bytes)
                        image_count += 1
                        data.add_field('images', image_bytes, filename=f'image{j}.jpg', content_type='image/jpeg')

                print(f"Batch {batch_number}: Sending {image_count} images, total size: {total_image_size/1024/1024:.2f}MB")

                async with session.post(QWEN_API_ANALYZE_URL, data=data) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        print(f"Error in batch {batch_number}: Status {response.status}")
                        print(f"Error body: {error_body}")
                        raise HTTPException(500, f"Analysis failed: {error_body}")
                    
                    print(f"Batch {batch_number}: Got response, parsing results")
                    results = await response.json()
                    if not isinstance(results, list):
                        results = [results]
                    
                    print(f"Batch {batch_number}: Processing {len(results)} results")
                    
                    for section, result in zip(current_batch, results):
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
                    
                    analyzed_sections.extend(current_batch)
                    print(f"Batch {batch_number}: Completed successfully")
                    
            except Exception as e:
                print(f"Critical error in batch {batch_number}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Current batch size: {len(current_batch)}")
                print(f"Total analyzed so far: {len(analyzed_sections)}")
                raise  # Re-raise the exception after logging
            
            await asyncio.sleep(0.1)
    
    print(f"Analysis completed. Processed {len(analyzed_sections)} sections in total")
    return analyzed_sections

async def process_sections(mask_result: Dict, normalized_prompt: str):
    section_map = build_section_map(mask_result["sections"])
    filtered_sections = []
    retry_count = 0
    max_retries = 1

    async with aiohttp.ClientSession() as session:
        while not filtered_sections and retry_count <= max_retries:
            for section in mask_result["sections"]:
                data = {
                    "normalized_prompt": normalized_prompt,
                    "sections": [{
                        "position_metadata": section["position_metadata"],
                        "image": section["image"]
                    }],
                    "relaxed": retry_count > 0
                }
                async with session.post(QWEN_API_FILTER_URL, json=data) as response:
                    if response.status != 200:
                        continue
                    result = await response.json()
                    if result["results"][0]["likely_contains"]:
                        filtered_sections.append(section)
            
            retry_count += 1

    sections_to_analyze = []
    for section in filtered_sections:
        sections_to_analyze.append(section)
        if section.get("has_children") and section.get("children"):
            for child in section["children"]:
                sections_to_analyze.append(child)
                if "neighbors" in child:
                    for direction in ["left", "right", "above", "below"]:
                        if child["neighbors"].get(direction):
                            neighbor_id = child["neighbors"][direction]
                            if neighbor_id in section_map:
                                sections_to_analyze.append(section_map[neighbor_id])

    analyzed_sections = await analyze_sections(sections_to_analyze)

    for section in analyzed_sections:
        section_map[section["id"]] = section

    resolve_children_neighbors(mask_result, section_map)

    return {
        "filtered_section_ids": [section["id"] for section in filtered_sections],
        "analyzed_section_ids": [section["id"] for section in analyzed_sections]
    }

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), prompt: str = Form(...),
    include_mask: bool = Query(False), debug: bool = Query(False)):
    
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
                print(f"Sections from mask-generation: {len(mask_result['sections'])}")
                
            await cache_collection.insert_one({
                "image_hash": image_hash,
                "result": mask_result,
                "created_at": datetime.utcnow()
            })

    async with aiohttp.ClientSession() as session:
        normalize_data = {"prompt": prompt}
        async with session.post(QWEN_API_NORMALIZE_URL, json=normalize_data) as response:
            if response.status != 200:
                raise HTTPException(500, "Prompt normalization failed")
            normalized_prompt = await response.json()

    process_result = await process_sections(mask_result, normalized_prompt)
    filtered_ids = process_result["filtered_section_ids"]
    analyzed_ids = process_result["analyzed_section_ids"]
    
    filtered_sections = [s for s in mask_result["sections"] if s["id"] in filtered_ids]
    children = await collect_children_for_matching(filtered_sections)
    
    initial_matches = await do_initial_matches(children, normalized_prompt)
    final_match = await reduce_matches(initial_matches, normalized_prompt)
    
    response = {
        "filtered_section_ids": filtered_ids,
        "children_count": len(children),
        "match": final_match
    }
        
    if debug:
        analyzed_sections = []
        for section in mask_result["sections"]:
            if section["id"] in analyzed_ids:
                analyzed_sections.append(prepare_section_for_json(section))
        response["debug"] = analyzed_sections
        
    if include_mask:
        response["mask_result"] = prepare_mask_result_for_json(mask_result)
        
    return response