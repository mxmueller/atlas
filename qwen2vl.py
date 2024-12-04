from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from vllm import LLM, SamplingParams
import uvicorn
import logging
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

app = FastAPI()

llm = LLM(
    model="Qwen/Qwen2-VL-72B-Instruct-AWQ", 
    trust_remote_code=True,
    dtype="float16",
    max_model_len=32768,
    enforce_eager=True,
    disable_custom_all_reduce=True,
    max_num_seqs=1,
)

@app.post("/generate")
async def generate_response(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Read image
        image_content = await image.read()
        
        # Convert to base64
        img_base64 = base64.b64encode(image_content).decode('utf-8')
        
        # Create prompt exactly as in the example
        prompt_text = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|>"
            "<|image_pad|>"  # Important: This tag is needed
            "<|vision_end|>\n"
            f"{prompt}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Create input with multimodal data
        inputs = {
            "prompt": prompt_text,
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
        
        return JSONResponse(content={"response": outputs[0].outputs[0].text})
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    try:
        dist.destroy_process_group()
    except:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000)