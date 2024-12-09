from typing import List, Dict, Tuple
import numpy as np

class LayoutAnalyzer:
    def __init__(self, min_gap_size: int = 20):
        self.min_gap_size = min_gap_size

    def analyze(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        y_coords = []
        for det in detections:
            y1, y2 = det['box'][1], det['box'][3]
            y_coords.extend([(y1, 'start'), (y2, 'end')])

        y_coords.sort(key=lambda x: x[0])
        
        current_open = 0
        gaps = []
        last_y = 0

        for y, coord_type in y_coords:
            if current_open == 0 and last_y > 0:
                gap_size = y - last_y
                if gap_size >= self.min_gap_size:
                    gaps.append((last_y, y))

            if coord_type == 'start':
                current_open += 1
            else:
                current_open -= 1
            last_y = y

        containers = []
        current_y = 0
        image_height = image_size[1]

        for gap_start, gap_end in gaps:
            if current_y < gap_start:
                elements = self._get_elements_in_section(detections, current_y, gap_start)
                if elements:
                    containers.append({
                        'id': f'section_{len(containers)}',
                        'box': [0, current_y, image_size[0], gap_start],
                        'elements': elements,
                        'type': 'section',
                        'confidence': 1.0,
                        'label': 'section'
                    })
            current_y = gap_end

        if current_y < image_height:
            elements = self._get_elements_in_section(detections, current_y, image_height)
            if elements:
                containers.append({
                    'id': f'section_{len(containers)}',
                    'box': [0, current_y, image_size[0], image_height],
                    'elements': elements,
                    'type': 'section',
                    'confidence': 1.0,
                    'label': 'section'
                })

        return containers

    def _get_elements_in_section(self, detections: List[Dict], y1: int, y2: int) -> List[Dict]:
        section_elements = []
        for det in detections:
            det_y1, det_y2 = det['box'][1], det['box'][3]
            element_height = det_y2 - det_y1
            overlap = min(det_y2, y2) - max(det_y1, y1)
            if overlap > element_height * 0.4:
                section_elements.append(det)
        return section_elements