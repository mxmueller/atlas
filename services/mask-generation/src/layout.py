import numpy as np
from scipy.ndimage import gaussian_filter, label
from typing import List, Dict, Tuple

class LayoutAnalyzer:
    def __init__(self):
        self.params = {
            'large': {
                'gaussian_sigma': 25,
                'density_threshold': 0.15,
                'min_area': 20000,
                'level': 1
            },
            'medium': {
                'gaussian_sigma': 20,
                'density_threshold': 0.2,
                'min_area': 5000,
                'level': 2
            },
            'small': {
                'gaussian_sigma': 15,
                'density_threshold': 0.25,
                'min_area': 1000,
                'level': 3
            }
        }
    
    def create_density_map(self, detections: List[Dict], image_size: Tuple[int, int], params: Dict) -> np.ndarray:
        density_map = np.zeros(image_size[::-1])
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            density_map[y1:y2, x1:x2] += 1
        
        density_map = gaussian_filter(density_map, sigma=params['gaussian_sigma'])
        if np.max(density_map) > 0:
            density_map = density_map / np.max(density_map)
        return density_map

    def find_containers(self, density_map: np.ndarray, params: Dict) -> List[Dict]:
        binary_map = density_map > params['density_threshold']
        labeled_array, num_features = label(binary_map)
        containers = []

        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            y_indices, x_indices = np.where(region_mask)
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
            x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))
            area = (x2 - x1) * (y2 - y1)

            if area < params['min_area']:
                continue

            containers.append({
                'box': [x1, y1, x2, y2],
                'level': params['level'],
                'area': area
            })

        return containers

    def find_parent(self, container: Dict, higher_level_containers: List[Dict]) -> str:
        x1, y1, x2, y2 = container['box']
        
        for parent in higher_level_containers:
            px1, py1, px2, py2 = parent['box']
            if x1 >= px1 and x2 <= px2 and y1 >= py1 and y2 <= py2:
                return parent.get('id', 'unknown')
        return 'root'

    def analyze(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        all_containers = []
        previous_level_containers = []
        
        for size, params in self.params.items():
            density_map = self.create_density_map(detections, image_size, params)
            containers = self.find_containers(density_map, params)
            
            for i, container in enumerate(containers):
                container['id'] = f"{size}_{i}"
                container['parent'] = self.find_parent(container, previous_level_containers)
                
            all_containers.extend(containers)
            previous_level_containers = containers
            
        return sorted(all_containers, key=lambda x: (-x['level'], x['area'], x['box'][1], x['box'][0]))