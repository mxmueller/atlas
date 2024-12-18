import requests
from io import BytesIO

def test_image_processing():
    response = requests.get('http://localhost:5000/random-screen')
    data = response.json()
    
    image_hex = data['image']
    prompt = data['data']['prompt']
    bbox = data['data']['bbox']
    
    image_bytes = BytesIO(bytes.fromhex(image_hex))
    files = {'file': ('image.png', image_bytes, 'image/png')}
    data = {'prompt': prompt}
    
    response = requests.post('http://localhost:9999/process-image', files=files, data=data)
    result = response.json()
    
    position = result['match']['position']
    threshold = 10
    
    assert abs(position[0] - bbox['x']) <= threshold
    assert abs(position[1] - bbox['y']) <= threshold