import pytest
import json
import requests
import time
import docker
from pathlib import Path
from io import BytesIO
from collections import defaultdict

client = docker.from_env()
containers = ['atlas_workflow-engine_1', 'atlas_mask-generation_1', 'atlas_qwen2-vl_1']

class TestStats:
   def __init__(self):
       self.stats = defaultdict(list)
       
@pytest.fixture(scope="session")
def test_stats():
   return TestStats()

def get_json_files():
   applied_dir = Path("applied")
   return list(applied_dir.glob("*.json"))

test_counter = 0

def wait_for_services():
   max_retries = 30
   retry_interval = 2
   
   for attempt in range(max_retries):
       try:
           response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
           if response.status_code == 200:
               return True
       except requests.exceptions.RequestException:
           pass
       time.sleep(retry_interval)
   
   raise Exception("Service nicht erreichbar nach Timeout")

@pytest.mark.parametrize('json_file', get_json_files())
def test_position(json_file, test_stats, tolerance_percent=10):
   global test_counter
   test_counter += 1
   
   if test_counter % 20 == 0:
       for container_name in containers:
           container = client.containers.get(container_name)
           container.restart()
       wait_for_services()
       
   start_time = time.time()
   
   with open(json_file) as f:
       data = json.load(f)
   
   with open(Path("applied") / data["image_path"], 'rb') as img:
       image_data = img.read()
       
   files = {
       'file': ('image.png', BytesIO(image_data), 'image/png'),
       'prompt': (None, data['prompt'])
   }
   
   response = requests.post('http://localhost:9999/process-image', files=files)
   result = response.json()
   
   x, y = result["match"]["position"]
   box = data["bounding_box"]
   
   width = box["x2"] - box["x1"] 
   height = box["y2"] - box["y1"]
   buffer_x = width * (tolerance_percent/100)
   buffer_y = height * (tolerance_percent/100)

   x_min = box["x1"] - buffer_x
   x_max = box["x2"] + buffer_x
   y_min = box["y1"] - buffer_y 
   y_max = box["y2"] + buffer_y

   x_deviation = min(abs(x - box["x1"]), abs(x - box["x2"])) / width * 100
   y_deviation = min(abs(y - box["y1"]), abs(y - box["y2"])) / height * 100

   duration = time.time() - start_time
   
   stats = {
       "file": json_file.name,
       "duration": duration,
       "x_deviation": x_deviation,
       "y_deviation": y_deviation,
       "response_time": response.elapsed.total_seconds()
   }
   test_stats.stats["tests"].append(stats)
   
   assert x_min <= x <= x_max and y_min <= y <= y_max, \
          f"Position ({x}, {y}) outside bounds {box} with {tolerance_percent}% tolerance"

def pytest_sessionfinish(session, test_stats):
   tests = test_stats.stats["tests"]
   print("\n=== Test Statistics ===")
   print(f"Total tests: {len(tests)}")
   print(f"Average duration: {sum(t['duration'] for t in tests)/len(tests):.2f}s")
   print(f"Average response time: {sum(t['response_time'] for t in tests)/len(tests):.2f}s")
   print(f"Average X deviation: {sum(t['x_deviation'] for t in tests)/len(tests):.1f}%")
   print(f"Average Y deviation: {sum(t['y_deviation'] for t in tests)/len(tests):.1f}%")
   print("\nSlowest tests:")
   for t in sorted(tests, key=lambda x: x['duration'], reverse=True)[:3]:
       print(f"- {t['file']}: {t['duration']:.2f}s")