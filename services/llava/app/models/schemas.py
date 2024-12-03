from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class UIElement(BaseModel):
    element_type: str
    visual_properties: Dict[str, Any]
    text: str
    matches_prompt: bool
    match_confidence: float
    match_reasoning: Optional[str]

class UIElementMatchInput(BaseModel):
    elements: List[Dict[str, Any]]
    query: str

class UIElementMatch(BaseModel):
    element: Dict[str, Any]
    confidence: float
    reasoning: str