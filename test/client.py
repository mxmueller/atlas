import requests
from PIL import Image
import io

def test_api(image_path, user_context=""):
    with open(image_path, 'rb') as img:
        files = {'image': ('image.jpg', img, 'image/jpeg')}
        data = {'user_context': user_context}
        
        response = requests.post(
            'http://localhost:8000/analyze',  # Geändert zu /analyze
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nAnalysis Result:")
            print(f"Type: {result.get('type', 'N/A')}")
            print(f"Text: {result.get('text', 'N/A')}")
            print(f"Visual Elements: {', '.join(result.get('visual_elements', ['N/A']))}")
            print(f"Primary Function: {result.get('primary_function', 'N/A')}")
            print(f"Dominant Color: {result.get('dominant_color', 'N/A')}")
        else:
            print(f"Error: Status Code {response.status_code}")
            print(f"Response: {response.json()}")

if __name__ == "__main__":
    # Beispiel ohne Kontext
    print("\nTesting:")
    test_api("./examples/next.png")
