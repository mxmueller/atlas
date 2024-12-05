import requests
from PIL import Image
import io
from typing import List, Union

def test_api(image_paths: Union[str, List[str]], user_context=""):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    

    files = []
    for i, path in enumerate(image_paths):
        with open(path, 'rb') as img:
            files.append(('images', ('image{}.jpg'.format(i), img.read(), 'image/jpeg')))
    
    data = {'user_context': user_context}
    
    response = requests.post(
        'http://localhost:8000/analyze',
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        results = response.json()

        if not isinstance(results, list):
            results = [results]
            
        for i, result in enumerate(results):
            print(f"\nAnalysis Result for image {i+1}:")
            print(f"Type: {result.get('type', 'N/A')}")
            print(f"Text: {result.get('text', 'N/A')}")
            print(f"Visual Elements: {', '.join(result.get('visual_elements', ['N/A']))}")
            print(f"Primary Function: {result.get('primary_function', 'N/A')}")
            print(f"Dominant Color: {result.get('dominant_color', 'N/A')}")
    else:
        print(f"Error: Status Code {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
   
    # Test mit mehreren Bildern
    print("\nTesting multiple images:")
    test_api([
        "./examples/next.png",
        "./examples/home.png",
        "./examples/radio.png",
        "./examples/store.png",
        "./examples/herunterladen.png"
    ])