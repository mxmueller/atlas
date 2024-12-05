class Settings:
    MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    MAX_MODEL_LEN = 32768
    MAX_NUM_BATCHED_TOKENS = 32768
    MAX_NUM_SEQS = 64
    WORKERS = 4
    HOST = "0.0.0.0"
    PORT = 8000

settings = Settings()