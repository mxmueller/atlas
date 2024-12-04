import requests
from PIL import Image
import io

def test_api(image_path, prompt):
    with open(image_path, 'rb') as img:
        files = {'image': ('image.jpg', img, 'image/jpeg')}
        data = {'prompt': prompt}
        
        response = requests.post(
            'http://localhost:8000/generate',
            files=files,
            data=data
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_api("./examples/home.png", "Describe this ui button?")