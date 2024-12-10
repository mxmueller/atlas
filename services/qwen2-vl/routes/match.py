from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional
from vllm import SamplingParams
from models.llm import LLMSingleton

router = APIRouter()

class UIElement(BaseModel):
    id: str
    type: str
    text: Optional[str]
    visual_elements: List[str]
    primary_function: str
    dominant_color: Optional[str]
    neighbours: Dict[str, Optional['UIElement']]

class PromptMatch(BaseModel):
    normalized_prompt: dict
    elements: List[UIElement]

def create_comparison_prompt(base_prompt: dict, element: UIElement) -> str:
    return (
        "<|im_start|>system\n"
        "Compare UI element with target description. Consider screen position and surrounding context.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Target UI element: {str(base_prompt)}\n\n"
        f"Current element to evaluate: {element.dict()}\n"
        f"Context - Neighboring elements:\n"
        f"Left: {element.neighbours.get('left')}\n"
        f"Right: {element.neighbours.get('right')}\n"
        f"Above: {element.neighbours.get('above')}\n"
        f"Below: {element.neighbours.get('below')}\n\n"
        "Rate how well this element matches the target description (0-100) considering:\n"
        "1. Element properties (type, color, text)\n"
        "2. Screen position matches requested position\n"
        "3. Context from surrounding elements\n\n"
        "Return only number 0-100\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

@router.post("/match")
async def match_elements(request: PromptMatch):
    try:
        llm_singleton = LLMSingleton()
        best_score = -1
        best_match_id = None
        
        prompts = [
            create_comparison_prompt(request.normalized_prompt, element)
            for element in request.elements
        ]
        
        async with llm_singleton._lock:
            outputs = llm_singleton.llm.generate(
                prompts=prompts,
                sampling_params=SamplingParams(
                    temperature=0.1,
                    max_tokens=32
                )
            )
            
            for idx, output in enumerate(outputs):
                try:
                    score = float(output.outputs[0].text.strip())
                    if score > best_score:
                        best_score = score
                        best_match_id = request.elements[idx].id
                except ValueError:
                    continue
        
        return {
            "match_id": best_match_id,
            "confidence_score": best_score
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }