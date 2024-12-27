from vllm import LLM
import asyncio
import torch
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings

class LLMSingleton:
   _instance = None
   _lock = asyncio.Lock()

   def __new__(cls):
       if cls._instance is None:
           cls._instance = super(LLMSingleton, cls).__new__(cls)
           cls._instance._initialize()
       return cls._instance
       
   def _initialize(self):
       self.llm = LLM(
           model=settings.MODEL_NAME,
           trust_remote_code=True,
           dtype="float16",
           max_model_len=settings.MAX_MODEL_LEN,
           enforce_eager=True,
           disable_custom_all_reduce=True,
           max_num_batched_tokens=settings.MAX_NUM_BATCHED_TOKENS,
           max_num_seqs=settings.MAX_NUM_SEQS
       )
       self.executor = ThreadPoolExecutor(max_workers=settings.WORKERS)

   def _get_detailed_memory_stats(self):
       stats = {
           'ram': {
               'total': psutil.virtual_memory().total / (1024**3),
               'used': psutil.virtual_memory().used / (1024**3),
               'percent': psutil.virtual_memory().percent
           }
       }
       
       if torch.cuda.is_available():
           memory_stats = torch.cuda.memory_stats()
           
           stats['cuda'] = {
               'allocated': torch.cuda.memory_allocated() / (1024**3),
               'reserved': torch.cuda.memory_reserved() / (1024**3),
               'max_allocated': torch.cuda.max_memory_allocated() / (1024**3),
               'max_reserved': torch.cuda.max_memory_reserved() / (1024**3),
               'non_releasable': (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / (1024**3)
           }
           
           for key in [
               'num_alloc_retries',
               'device_fragmentation',
               'active_bytes.all.current',
               'inactive_split_bytes.all.current',
               'reserved_bytes.all.current',
               'active_blocks.all.current',
               'inactive_blocks.all.current',
               'allocated_bytes.all.current'
           ]:
               try:
                   value = memory_stats.get(key)
                   if value is not None:
                       if 'bytes' in key:
                           value = value / (1024**3)
                       stats['cuda'][key.replace('.all.current', '')] = value
               except:
                   pass

           if hasattr(self, 'llm') and hasattr(self.llm, 'engine'):
               try:
                   stats['vllm'] = {
                       'gpu_memory': self.llm.engine.worker.gpu_memory / (1024**3),
                       'gpu_memory_utilization': self.llm.engine.worker.gpu_memory_utilization,
                       'max_num_batched_tokens': self.llm.engine.scheduler.max_num_batched_tokens,
                       'max_num_seqs': self.llm.engine.scheduler.max_num_seqs
                   }
               except:
                   pass

       return stats

   async def reset(self):
       async with self._lock:
           try:
               before_stats = self._get_detailed_memory_stats()
               print(f"Memory before reset: {before_stats}")

               if hasattr(self, 'llm'):
                   if hasattr(self.llm, 'engine'):
                       try:
                           self.llm.engine.cache_manager.reset()
                       except:
                           pass

               if hasattr(self, 'executor'):
                   self.executor.shutdown()

               torch.cuda.empty_cache()
               gc.collect()

               if hasattr(self, 'llm'):
                   del self.llm

               self._initialize()
               
               after_stats = self._get_detailed_memory_stats()
               print(f"Memory after reset: {after_stats}")
               
               return {
                   'before': before_stats,
                   'after': after_stats
               }
               
           except Exception as e:
               print(f"Error during reset: {str(e)}")
               return {
                   'error': str(e),
                   'type': type(e).__name__
               }

   async def process_request(self, prompts, sampling_params):
       return await self.llm.generate(prompts, sampling_params)