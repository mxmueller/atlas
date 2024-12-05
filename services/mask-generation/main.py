import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Dict
import numpy as np

class RefinedUIDetector:
    def __init__(self):
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        
        # Optimierte Texterkennungs-Parameter
        self.text_detection_params = {
            'min_text_length': 2,          # Kürzere Minimallänge für Menüeinträge
            'max_text_gap': 30,            # Größerer Abstand für Menüpunkte
            'line_height_tolerance': 8,     # Größere Toleranz für Zeilenhöhe
            'word_spacing': 15,            # Angepasster Wortabstand
            'menu_item_max_gap': 40,       # Speziell für Menüleisten
            'paragraph_line_spacing': 5,    # Für Fließtext
            'list_item_indent': 20,        # Für Aufzählungen
            'heading_min_height': 25       # Für Überschriften
        }
        
        # Erweiterte Layout-Erkennung
        self.layout_patterns = {
            'menu_bar': {
                'height_range': (20, 40),
                'min_items': 3,
                'max_gap': 40
            },
            'paragraph': {
                'min_width': 200,
                'min_lines': 2,
                'line_spacing': 5
            },
            'list': {
                'indent': 20,
                'bullet_width': 15,
                'item_spacing': 8
            }
        }
        
        # Verbesserte Prompts
        self.prompts = [
            # Menüleisten
            ("a horizontal menu item text. a navigation bar text. " +
             "a top menu text. a menu item label. a menu bar item."),
            
            # Fließtext
            ("a paragraph text. a continuous text block. " +
             "a multi-line text. a text content block. " +
             "an article text. a description text."),
            
            # Listen und Aufzählungen
            ("a bullet point text. a list item text. " +
             "a numbered item text. an indented text. " +
             "a list entry text."),
            
            # Buttons und Links
            ("a button text. a link text. a clickable text. " +
             "an action text. a menu button text."),
            
            # Labels und Überschriften
            ("a heading text. a title text. a label text. " +
             "a caption text. a section text.")
        ]

    def detect_with_prompt(self, image, prompt, confidence_threshold):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        return results

    def detect(self, image_path: str, confidence_threshold: float = 0.12) -> tuple:
        try:
            print("Loading image...")
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None
        
        try:
            print("Starting detection...")
            all_detections = []
            
            for prompt in self.prompts:
                results = self.detect_with_prompt(image, prompt, confidence_threshold)
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= confidence_threshold:
                        all_detections.append({
                            'box': box.tolist(),
                            'score': score.item(),
                            'label': label
                        })
            
            # Verarbeite Layout
            processed_detections = self.process_layout(all_detections)
            
            print(f"Found {len(processed_detections)} UI elements")
            return processed_detections, image
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return None, None

    def process_layout(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
        
        # Entferne zu kleine/große Detektionen
        filtered = self.filter_by_size(detections)
        
        # Gruppiere nach Layout-Typ
        groups = self.group_by_layout(filtered)
        
        # Verarbeite jede Gruppe
        processed = []
        processed.extend(self.process_menu_items(groups['menu']))
        processed.extend(self.process_text_blocks(groups['text']))
        processed.extend(self.process_list_items(groups['list']))
        processed.extend(groups['other'])
        
        return processed

    def filter_by_size(self, detections: List[Dict]) -> List[Dict]:
        filtered = []
        for det in detections:
            width = det['box'][2] - det['box'][0]
            height = det['box'][3] - det['box'][1]
            area = width * height
            
            # Ignoriere zu große oder zu kleine Boxen
            if 50 < area < 50000:
                filtered.append(det)
        return filtered

    def group_by_layout(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        groups = {
            'menu': [],
            'text': [],
            'list': [],
            'other': []
        }
        
        for det in detections:
            height = det['box'][3] - det['box'][1]
            width = det['box'][2] - det['box'][0]
            
            # Menüelement
            if (self.layout_patterns['menu_bar']['height_range'][0] <= height <= 
                self.layout_patterns['menu_bar']['height_range'][1]):
                groups['menu'].append(det)
            
            # Textblock
            elif width >= self.layout_patterns['paragraph']['min_width']:
                groups['text'].append(det)
            
            # Listenelement
            elif det['box'][0] >= self.layout_patterns['list']['indent']:
                groups['list'].append(det)
            
            # Sonstiges
            else:
                groups['other'].append(det)
        
        return groups

    def process_menu_items(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        
        # Sortiere nach y-Position (Zeile) und dann x-Position
        items.sort(key=lambda x: (x['box'][1], x['box'][0]))
        
        processed = []
        current_line = []
        last_y = None
        
        for item in items:
            current_y = item['box'][1]
            
            if last_y is not None and abs(current_y - last_y) > self.text_detection_params['line_height_tolerance']:
                if current_line:
                    processed.extend(self.merge_menu_line(current_line))
                current_line = []
            
            current_line.append(item)
            last_y = current_y
        
        if current_line:
            processed.extend(self.merge_menu_line(current_line))
        
        return processed

    def merge_menu_line(self, line_items: List[Dict]) -> List[Dict]:
        if not line_items:
            return []
        
        # Sortiere nach x-Position
        line_items.sort(key=lambda x: x['box'][0])
        merged = []
        current_group = [line_items[0]]
        
        for item in line_items[1:]:
            last_item = current_group[-1]
            gap = item['box'][0] - last_item['box'][2]
            
            if gap < self.text_detection_params['menu_item_max_gap']:
                current_group.append(item)
            else:
                if current_group:
                    merged.append(self.merge_items(current_group))
                current_group = [item]
        
        if current_group:
            merged.append(self.merge_items(current_group))
        
        return merged

    def process_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        if not blocks:
            return []
        
        # Sortiere nach Position
        blocks.sort(key=lambda x: (x['box'][1], x['box'][0]))
        
        merged = []
        current_block = []
        
        for block in blocks:
            if not current_block:
                current_block = [block]
            else:
                last_block = current_block[-1]
                vertical_gap = block['box'][1] - last_block['box'][3]
                
                if vertical_gap <= self.layout_patterns['paragraph']['line_spacing']:
                    current_block.append(block)
                else:
                    merged.append(self.merge_items(current_block))
                    current_block = [block]
        
        if current_block:
            merged.append(self.merge_items(current_block))
        
        return merged

    def process_list_items(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        
        # Sortiere nach y-Position
        items.sort(key=lambda x: x['box'][1])
        processed = []
        current_list = []
        
        for item in items:
            if not current_list:
                current_list = [item]
            else:
                last_item = current_list[-1]
                spacing = item['box'][1] - last_item['box'][3]
                
                if spacing <= self.layout_patterns['list']['item_spacing']:
                    current_list.append(item)
                else:
                    processed.extend(current_list)
                    current_list = [item]
        
        if current_list:
            processed.extend(current_list)
        
        return processed

    def merge_items(self, items: List[Dict]) -> Dict:
        if not items:
            return None
        
        # Berechne gemeinsame Box
        min_x = min(item['box'][0] for item in items)
        min_y = min(item['box'][1] for item in items)
        max_x = max(item['box'][2] for item in items)
        max_y = max(item['box'][3] for item in items)
        
        # Verwende höchsten Score
        max_score = max(item['score'] for item in items)
        
        # Kombiniere Labels
        labels = [item['label'] for item in items]
        combined_label = ' '.join(labels)
        
        return {
            'box': [min_x, min_y, max_x, max_y],
            'score': max_score,
            'label': combined_label
        }

    def visualize_results(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Farbschema
        colors = {
            'menu': (52, 152, 219),    # Blau
            'text': (46, 204, 113),    # Grün
            'list': (155, 89, 182),    # Lila
            'button': (231, 76, 60),   # Rot
            'label': (241, 196, 15),   # Gelb
            'default': (149, 165, 166)  # Grau
        }
        
        for det in detections:
            box = det['box']
            label = det['label'].lower()
            score = det['score']
            
            # Wähle Farbe
            color = colors['default']
            for key in colors:
                if key in label:
                    color = colors[key]
                    break
            
            # Zeichne Box
            draw.rectangle(box, outline=color, width=2)
            
            # Text mit Hintergrund
            text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((box[0], box[1]-20), text)
            draw.rectangle((text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3),
                         fill=(0, 0, 0))
            draw.text((box[0], box[1]-20), text, fill=color)
        
        return draw_image

def main():
    detector = RefinedUIDetector()
    
    image_path = "ard.png"
    print(f"Starting detection on {image_path}")
    
    detections, image = detector.detect(image_path)
    
    if detections and image:
        print("\nDetected UI Elements:")
        for det in detections:
            print(f"Element: {det['label']:<30} Confidence: {det['score']:.3f}")
        
        result_image = detector.visualize_results(image, detections)
        output_path = "detected_ui_elements_refined.png"
        result_image.save(output_path)
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("No UI elements detected or error during processing")

if __name__ == "__main__":
    main()