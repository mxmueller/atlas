from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Optional
import uvicorn

app = FastAPI(title="Qwen API Service")

# Initialize the model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=512
)
llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
    dtype="float16",
    quantization="gptq"
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 512
    repetition_penalty: Optional[float] = 1.05

class ChatResponse(BaseModel):
    generated_text: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Update sampling parameters if provided
        current_sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty
        )
        
        # Convert messages to the format expected by the tokenizer
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        outputs = llm.generate([text], current_sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return ChatResponse(generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)