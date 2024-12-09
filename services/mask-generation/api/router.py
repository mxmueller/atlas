from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image
import io
import base64
from src.detector import RefinedUIDetector
import uuid
import time
import hashlib

router = APIRouter()
detector = RefinedUIDetector()

def get_image_hash(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format='PNG')
    return hashlib.md5(data.getvalue()).hexdigest()[:8]

def generate_unique_id(prefix: str, image_hash: str) -> str:
    timestamp = int(time.time() * 1000)
    unique = str(uuid.uuid4())[:8]
    return f"{prefix}_{image_hash}_{timestamp}_{unique}"

def get_cropped_image_base64(image: Image.Image, box: list) -> str:
    cropped = image.crop(box)
    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_position(box: list) -> list:
    x1, y1, x2, y2 = box
    return [int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.25)]

def get_section_metadata(box: list, image_height: int) -> dict:
    y_start = (box[1] / image_height) * 100
    y_end = (box[3] / image_height) * 100
    return {
        "y_start": round(y_start, 2),
        "y_end": round(y_end, 2),
        "vertical_position": "top" if y_start < 33 else "middle" if y_start < 66 else "bottom"
    }

def calculate_neighbors(elements: list) -> dict:
    neighbor_map = {}
    for i, elem in enumerate(elements):
        box1 = elem['box']
        neighbors = {
            "left": None,
            "right": None,
            "above": None,
            "below": None
        }
        
        for other in elements:
            if elem['id'] == other['id']:
                continue
            box2 = other['box']
            
            if (box1[1] < box2[3] and box1[3] > box2[1]):
                if box2[2] < box1[0]:
                    if (not neighbors["left"] or 
                        box1[0] - box2[2] < box1[0] - elements[next(i for i, e in enumerate(elements) if e['id'] == neighbors["left"])]['box'][2]):
                        neighbors["left"] = other['id']
                elif box2[0] > box1[2]:
                    if (not neighbors["right"] or 
                        box2[0] - box1[2] < elements[next(i for i, e in enumerate(elements) if e['id'] == neighbors["right"])]['box'][0] - box1[2]):
                        neighbors["right"] = other['id']
            
            if (box1[0] < box2[2] and box1[2] > box2[0]):
                if box2[3] < box1[1]:
                    if (not neighbors["above"] or 
                        box1[1] - box2[3] < box1[1] - elements[next(i for i, e in enumerate(elements) if e['id'] == neighbors["above"])]['box'][3]):
                        neighbors["above"] = other['id']
                elif box2[1] > box1[3]:
                    if (not neighbors["below"] or 
                        box2[1] - box1[3] < elements[next(i for i, e in enumerate(elements) if e['id'] == neighbors["below"])]['box'][1] - box1[3]):
                        neighbors["below"] = other['id']
        
        neighbor_map[elem['id']] = neighbors
    
    return neighbor_map

@router.post("/api/mask")
async def create_mask(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
        
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    ui_detections, text_detections, layout_containers, _ = detector.detect(image)
    if not ui_detections and not text_detections:
        raise HTTPException(500, "Processing failed")
        
    result_image = detector.visualizer.visualize_results(
        image, 
        ui_detections + text_detections,
        layout_containers
    )
    
    total_width = image.width * 2
    max_height = image.height
    comparison = Image.new('RGB', (total_width, max_height))
    comparison.paste(image, (0, 0))
    comparison.paste(result_image, (image.width, 0))
    
    img_byte_arr = io.BytesIO()
    comparison.save(img_byte_arr, format='PNG')
    
    return Response(
        content=img_byte_arr.getvalue(),
        media_type="image/png"
    )

@router.post("/api/artifacts")
async def extract_artifacts(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
        
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_hash = get_image_hash(image)
    
    ui_detections, text_detections, layout_containers, _ = detector.detect(image)
    
    all_elements = []
    sections = []
    
    for container in layout_containers:
        section_id = generate_unique_id('section', image_hash)
        section_box = [int(x) for x in container['box']]
        section_elements = []
        
        for element in container['elements']:
            element_box = [int(x) for x in element['box']]
            element_id = generate_unique_id('element', image_hash)
            
            element_data = {
                "id": element_id,
                "score": element['score'],
                "label": element['label'],
                "box": element_box,
                "position": get_position(element_box),
                "image": get_cropped_image_base64(image, element_box),
                "section_id": section_id
            }
            
            all_elements.append(element_data)
            section_elements.append(element_data)
        
        sections.append({
            "id": section_id,
            "box": section_box,
            "image": get_cropped_image_base64(image, section_box),
            "position_metadata": get_section_metadata(section_box, image.size[1]),
            "num_children": len(section_elements),
            "children": section_elements
        })
    
    neighbors = calculate_neighbors(all_elements)
    
    for section in sections:
        for element in section['children']:
            element["neighbors"] = neighbors[element["id"]]
    
    return JSONResponse(content={"sections": sections})