from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from vllm import LLM, SamplingParams
import uvicorn
import logging
import torch.distributed as dist
import json

logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class LLMSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMSingleton, cls).__new__(cls)
            cls._instance.llm = LLM(
                model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
                trust_remote_code=True,
                dtype="float16",
                max_model_len=32768,
                enforce_eager=True,
                disable_custom_all_reduce=True,
            )
        return cls._instance

app = FastAPI()
llm = LLMSingleton().llm

def create_analysis_prompt() -> str:
    base_prompt = (
        "<|im_start|>system\n"
        "You are a precise UI element analyzer. Extract information ONLY from what you can see.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Example output format:\n"
        "{\n"
        '    "type": "button|icon|text|input",\n'
        '    "text": "exact text if present, null if none",\n'
        '    "visual_elements": ["icon names or descriptions if present else put none"],\n'
        '    "primary_function": "main purpose based on visual evidence only make two sentence",\n'
        '    "dominant_color": "main color if clearly visible, null if unclear"\n'
        "}\n\n"
    )
    
    return (
        f"{base_prompt}"
        "<|vision_start|>"
        "<|image_pad|>"
        "<|vision_end|>\n"
        "Analyze this UI element and return valid JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

@app.post("/analyze")
async def analyze_ui_element(
    image: UploadFile = File(...)
):
    try:
        image_content = await image.read()
        
        inputs = {
            "prompt": create_analysis_prompt(),
            "multi_modal_data": {
                "image": Image.open(io.BytesIO(image_content)).convert("RGB")
            }
        }
        
        outputs = llm.generate(
            inputs,
            sampling_params=SamplingParams(
                temperature=0.2,
                max_tokens=512
            )
        )

        try:
            response_text = outputs[0].outputs[0].text
            parsed_json = json.loads(response_text)
            return JSONResponse(content=parsed_json)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "Invalid JSON response", "raw_response": response_text})
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    try:
        dist.destroy_process_group()
    except:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000)