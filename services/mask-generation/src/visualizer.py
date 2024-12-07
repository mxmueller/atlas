from PIL import Image, ImageDraw
from typing import List, Dict
from config.settings import VISUALIZATION_COLORS

class UIVisualizer:
    def visualize_results(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        for det in detections:
            box = det['box']
            label = det['label'].lower()
            score = det['score']
            
            color = VISUALIZATION_COLORS['default']
            for key in VISUALIZATION_COLORS:
                if key in label:
                    color = VISUALIZATION_COLORS[key]
                    break
            
            draw.rectangle(box, outline=color, width=2)
            
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]-20), text)
            draw.rectangle((text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3),
                         fill=(0, 0, 0))
            draw.text((box[0], box[1]-20), text, fill=color)
        
        return draw_image

    def save_visualization(self, image: Image.Image, output_path: str):
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")