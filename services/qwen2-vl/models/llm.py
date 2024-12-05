from vllm import LLM
import asyncio
from concurrent.futures import ThreadPoolExecutor
from config.settings import settings  # Relativer Import entfernt

class LLMSingleton:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMSingleton, cls).__new__(cls)
            cls._instance.llm = LLM(
                model=settings.MODEL_NAME,
                trust_remote_code=True,
                dtype="float16",
                max_model_len=settings.MAX_MODEL_LEN,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                max_num_batched_tokens=settings.MAX_NUM_BATCHED_TOKENS,
                max_num_seqs=settings.MAX_NUM_SEQS
            )
            cls._instance.executor = ThreadPoolExecutor(max_workers=settings.WORKERS)
        return cls._instance