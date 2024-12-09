from fastapi import FastAPI, File, UploadFile, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import hashlib
from io import BytesIO
import logging
from datetime import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = AsyncIOMotorClient("mongodb://mongo:27017")
db = mongo_client.cache_db
cache_collection = db.image_cache

MASK_API_URL = "http://mask-generation:8000/api/artifacts"

async def get_image_hash(image_bytes: bytes) -> str:
  return hashlib.md5(image_bytes).hexdigest()

@app.on_event("startup")
async def startup_event():
   await cache_collection.create_index(
       "created_at", 
       expireAfterSeconds=43200  # 12 Stunden
   )

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
  if not file.content_type.startswith('image/'):
      raise HTTPException(400, "File must be an image")
  
  content = await file.read()
  image_hash = await get_image_hash(content)
  
  cached_result = await cache_collection.find_one({"image_hash": image_hash})
  if cached_result:
      logger.info(f"Cache HIT for hash: {image_hash}")
      return cached_result["result"]
      
  logger.info(f"Cache MISS for hash: {image_hash}")
  async with aiohttp.ClientSession() as session:
      form = aiohttp.FormData()
      form.add_field('file', 
                    BytesIO(content), 
                    filename=file.filename,
                    content_type=file.content_type)
      
      async with session.post(MASK_API_URL, data=form) as response:
          if response.status != 200:
              raise HTTPException(500, "Mask generation failed")
          result = await response.json()
          
  await cache_collection.insert_one({
      "image_hash": image_hash,
      "result": result,
      "created_at": datetime.utcnow()
  })
  
  return result