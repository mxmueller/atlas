from vllm import LLM
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings
import torch
import gc
import psutil
import os

class LLMSingleton:
   _instance = None
   _lock = asyncio.Lock()

   def __new__(cls):
       if cls._instance is None:
           cls._instance = super(LLMSingleton, cls).__new__(cls)
           cls._instance._init()
       return cls._instance

   def _init(self):
       print("\n" + "="*50)
       print("🚀 Initializing vLLM")
       print(f"💾 Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
       
       self.llm = LLM(
           model=settings.MODEL_NAME,
           trust_remote_code=True,
           dtype="float16", 
           max_model_len=settings.MAX_MODEL_LEN,
           enforce_eager=True,
           disable_custom_all_reduce=True,
           max_num_batched_tokens=settings.MAX_NUM_BATCHED_TOKENS,
           max_num_seqs=settings.MAX_NUM_SEQS,
           gpu_memory_utilization=0.85
       )
       self.executor = ThreadPoolExecutor(max_workers=settings.WORKERS)
       
       print(f"📊 Initial Memory Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
       print(f"📊 Initial Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
       print("="*50 + "\n")

   async def process_request(self, *args, **kwargs):
       # Memory vor der Inferenz
       before_reserved = torch.cuda.memory_reserved()/1024**3
       before_allocated = torch.cuda.memory_allocated()/1024**3
       
       print("\n" + "="*50)
       print("📊 MEMORY BEFORE INFERENCE:")
       print(f"Reserved: {before_reserved:.2f} GB")
       print(f"Allocated: {before_allocated:.2f} GB")
       
       # Inferenz
       result = self.llm.generate(*args, **kwargs)
       
       # Memory Cleanup nach Inferenz
       torch.cuda.synchronize()
       gc.collect()
       torch.cuda.empty_cache()
       torch.cuda.reset_peak_memory_stats()
       
       # Memory nach Cleanup
       after_reserved = torch.cuda.memory_reserved()/1024**3
       after_allocated = torch.cuda.memory_allocated()/1024**3
       
       print("\n📊 MEMORY AFTER INFERENCE & CLEANUP:")
       print(f"Reserved: {after_reserved:.2f} GB")
       print(f"Allocated: {after_allocated:.2f} GB")
       print(f"Diff Reserved: {after_reserved - before_reserved:.2f} GB")
       print(f"Diff Allocated: {after_allocated - before_allocated:.2f} GB")
       print("="*50 + "\n")
       
       return result

   async def __aenter__(self):
       return self
       
   async def __aexit__(self, exc_type, exc_val, exc_tb):
       self.executor.shutdown(wait=True)