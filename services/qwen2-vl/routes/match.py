from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional
from vllm import SamplingParams
from models.llm import LLMSingleton

router = APIRouter()

class NeighborElement(BaseModel):
   id: str
   type: str  
   visual_elements: List[str]
   dominant_color: Optional[str]
   text: Optional[str] = None

   class Config:
       def dict(self, *args, **kwargs):
           data = super().dict(*args, **kwargs)
           return {k:v for k,v in data.items() if v is not None}

class UIElement(BaseModel):
   id: str
   type: str  
   visual_elements: List[str]
   dominant_color: Optional[str]
   text: Optional[str] = None
   primary_function: Optional[str] = None
   neighbors: Optional[Dict[str, Optional[NeighborElement]]] = None

   class Config:
       def dict(self, *args, **kwargs):
           data = super().dict(*args, **kwargs)
           return {k:v for k,v in data.items() 
                  if v is not None and 
                  (k != 'neighbors' or any(n is not None for n in v.values()))}

class PromptMatch(BaseModel):
   normalized_prompt: dict
   elements: List[UIElement]

def create_comparison_prompt(base_prompt: dict, elements: List[UIElement]) -> str:
   elements_processed = [e.dict() for e in elements]
   
   # Format target description
   target_formatted = "\n".join(f"- {k}: {v}" for k,v in base_prompt.items())
   
   # Format elements
   elements_formatted = ""
   for e in elements_processed:
       elements_formatted += f"\nElement:\n"
       for k,v in e.items():
           if k == 'neighbors':
               elements_formatted += f"- {k}:\n"
               for pos, n in v.items():
                   elements_formatted += f"  {pos}:\n"
                   for nk, nv in n.items():
                       elements_formatted += f"    {nk}: {nv}\n"
           else:
               elements_formatted += f"- {k}: {v}\n"
   
   prompt = (
       "<|im_start|>system\n"
       "You are a precise UI element matching system. Your task is to find the exact element that best matches "
       "the target description. Pay special attention to:\n"
       "- Exact type matches (button, text, icon etc)\n"
       "- Visual elements and their specific descriptions\n"
       "- Text content and phrasing\n"
       "- Color nuances\n"
       "- Primary function and purpose\n"
       "- Contextual placement if neighbors exist\n\n"
       "Analyze each element thoroughly before deciding. Return ONLY the id of the best matching element.\n"
       "<|im_end|>\n"
       "<|im_start|>user\n"
       f"Target Description:\n{target_formatted}\n\n"
       f"Available Elements:{elements_formatted}\n\n"
       "Which element id matches the target description most precisely?\n"
       "<|im_end|>\n"
       "<|im_start|>assistant\n"
   )
   
   print("\nDebug - Complete Prompt:")
   print(prompt)
   return prompt

@router.post("/match")
async def match_elements(request: PromptMatch):
  try:
      llm_singleton = LLMSingleton()
      prompt = create_comparison_prompt(
          request.normalized_prompt,
          request.elements
      )
      
      async with llm_singleton._lock:
          output = await llm_singleton.process_request(
              prompts=[prompt],
              sampling_params=SamplingParams(
                  temperature=0.1,
                  max_tokens=32
              )
          )
          
          raw_match_id = output[0].outputs[0].text.strip()
          print(f"\nRaw LLM output: {raw_match_id}")
          
          if "none" in raw_match_id.lower() or "no match" in raw_match_id.lower():
              return {"match_id": False}
              
          match_id = raw_match_id.replace('"','').replace("'","").strip()
          
          if not any(e.id == match_id for e in request.elements):
              return {"match_id": False}
              
          return {"match_id": match_id}
          
  except Exception as e:
      return {
          "error": str(e),
          "type": type(e).__name__
      }