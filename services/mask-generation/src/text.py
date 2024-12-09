import easyocr
from PIL import Image
import numpy as np

class TextDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'de'])
    
    def detect(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
            
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        
        detections = []
        for box, text, conf in results:
            # Convert box points to [x1,y1,x2,y2] format
            x1, y1 = min(point[0] for point in box), min(point[1] for point in box)
            x2, y2 = max(point[0] for point in box), max(point[1] for point in box)
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'score': conf,
                'label': text
            })
            
        return detections