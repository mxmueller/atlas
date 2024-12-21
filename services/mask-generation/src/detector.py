import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config.settings import MODEL_CONFIG, PROMPTS
from .processors import LayoutProcessor
from .visualizer import UIVisualizer
from .text import TextDetector
from .layout import LayoutAnalyzer

class RefinedUIDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_CONFIG['model_id'])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_CONFIG['model_id']).to(self.device)
        self.layout_processor = LayoutProcessor()
        self.visualizer = UIVisualizer()
        self.text_detector = TextDetector()
        self.layout_analyzer = LayoutAnalyzer()

    def detect(self, image: Image.Image, confidence_threshold: float = 0.15):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            ui_detections = []
            for prompt in PROMPTS:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                
                try:
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=confidence_threshold,
                        text_threshold=confidence_threshold,
                        target_sizes=[image.size[::-1]]
                    )[0]
                    
                    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                        if score >= confidence_threshold:
                            ui_detections.append({
                                'box': box.tolist(),
                                'score': score.item(),
                                'label': label
                            })
                            
                finally:
                    del inputs
                    if torch.cuda.is_initialized():
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass

            text_detections = self.text_detector.detect(image)
            processed_ui = self.layout_processor.process_layout(ui_detections)
            all_detections = processed_ui + text_detections
            layout_containers = self.layout_analyzer.analyze(all_detections, image.size)
            
            return processed_ui, text_detections, layout_containers, image

        finally:
            if torch.cuda.is_initialized():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass

    def __del__(self):
        if hasattr(self, 'model'):
            try:
                self.model.cpu()
                del self.model
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            except:
                pass