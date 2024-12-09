from PIL import Image, ImageDraw
from typing import List, Dict

class UIVisualizer:
    def visualize_results(self, image: Image.Image, detections: List[Dict], containers: List[Dict] = None) -> Image.Image:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image, 'RGBA')
        
        if containers:
            for container in containers:
                box = container['box']
                draw.rectangle(box, fill=(100, 100, 255, 50), outline=(100, 100, 255, 200), width=2)
        
        for det in detections:
            box = det['box']
            label = det['label'].lower()
            score = det['score']
            
            draw.rectangle(box, outline=(255, 100, 100), width=2)
            
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]-20), text)
            draw.rectangle((text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3),
                         fill=(0, 0, 0))
            draw.text((box[0], box[1]-20), text, fill=(255, 100, 100))
        
        return draw_image

    def save_visualization(self, image: Image.Image, output_path: str):
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")