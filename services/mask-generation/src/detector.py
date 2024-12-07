import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Dict, Tuple
from config.settings import MODEL_CONFIG, PROMPTS
from .processors import LayoutProcessor
from .visualizer import UIVisualizer
from .text import TextDetector
from .layout import LayoutAnalyzer

class RefinedUIDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(MODEL_CONFIG['model_id'])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_CONFIG['model_id']).to(self.device)
        
        self.layout_processor = LayoutProcessor()
        self.visualizer = UIVisualizer()
        self.text_detector = TextDetector()
        self.layout_analyzer = LayoutAnalyzer()

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

    def detect(self, image_path: str, confidence_threshold: float = 0.08) -> Tuple[List[Dict], List[Dict], List[Dict], Image.Image]:
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None, None, None
        
        try:
            print("Starting UI detection...")
            ui_detections = []
            
            for prompt in PROMPTS:
                results = self.detect_with_prompt(image, prompt, confidence_threshold)
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= confidence_threshold:
                        ui_detections.append({
                            'box': box.tolist(),
                            'score': score.item(),
                            'label': label
                        })
            
            print("Starting text detection...")
            text_detections = self.text_detector.detect(image)
            
            processed_ui = self.layout_processor.process_layout(ui_detections)
            
            all_detections = processed_ui + text_detections
            layout_containers = self.layout_analyzer.analyze(all_detections, image.size)
            
            print(f"Found {len(processed_ui)} UI elements, {len(text_detections)} text elements, and {len(layout_containers)} layout containers")
            return processed_ui, text_detections, layout_containers, image
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return None, None, None, None

    def process_and_visualize(self, image_path: str, output_path: str, confidence_threshold: float = 0.12):
        ui_detections, text_detections, layout_containers, image = self.detect(image_path, confidence_threshold)
        
        if ui_detections and text_detections and image:
            all_detections = ui_detections + text_detections
            
            print("\nDetected Elements:")
            for det in all_detections:
                print(f"Element: {det['label']:<30} Confidence: {det['score']:.3f}")
            
            result_image = self.visualizer.visualize_results(image, all_detections, layout_containers)
            self.visualizer.save_visualization(result_image, output_path)