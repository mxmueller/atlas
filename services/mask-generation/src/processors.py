from typing import List, Dict
from config.settings import TEXT_DETECTION_PARAMS, LAYOUT_PATTERNS

class LayoutProcessor:
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)

    def merge_overlapping_boxes(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        if not detections:
            return []
            
        merged = []
        used = set()

        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            current_group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j in used or i == j:
                    continue
                    
                if self.calculate_iou(det1['box'], det2['box']) > iou_threshold:
                    current_group.append(det2)
                    used.add(j)
            
            if current_group:
                best_score_idx = max(range(len(current_group)), 
                                   key=lambda x: current_group[x]['score'])
                best_det = current_group[best_score_idx]
                unique_labels = []
                for d in current_group:
                    if d['label'] not in unique_labels:
                        unique_labels.append(d['label'])
                
                merged.append({
                    'box': best_det['box'],
                    'score': best_det['score'],
                    'label': ' | '.join(unique_labels)
                })
        
        return merged

    def process_layout(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
            
        detections = self.merge_overlapping_boxes(detections)
        filtered = self.filter_by_size(detections)
        groups = self.group_by_layout(filtered)
        
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
            
            if (LAYOUT_PATTERNS['menu_bar']['height_range'][0] <= height <= 
                LAYOUT_PATTERNS['menu_bar']['height_range'][1]):
                groups['menu'].append(det)
            elif width >= LAYOUT_PATTERNS['paragraph']['min_width']:
                groups['text'].append(det)
            elif det['box'][0] >= LAYOUT_PATTERNS['list']['indent']:
                groups['list'].append(det)
            else:
                groups['other'].append(det)
        
        return groups

    def process_menu_items(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        
        items.sort(key=lambda x: (x['box'][1], x['box'][0]))
        processed = []
        current_line = []
        last_y = None
        
        for item in items:
            current_y = item['box'][1]
            
            if last_y is not None and abs(current_y - last_y) > TEXT_DETECTION_PARAMS['line_height_tolerance']:
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
        
        line_items.sort(key=lambda x: x['box'][0])
        merged = []
        current_group = [line_items[0]]
        
        for item in line_items[1:]:
            last_item = current_group[-1]
            gap = item['box'][0] - last_item['box'][2]
            
            if gap < TEXT_DETECTION_PARAMS['menu_item_max_gap']:
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
        
        blocks.sort(key=lambda x: (x['box'][1], x['box'][0]))
        merged = []
        current_block = []
        
        for block in blocks:
            if not current_block:
                current_block = [block]
            else:
                last_block = current_block[-1]
                vertical_gap = block['box'][1] - last_block['box'][3]
                
                if vertical_gap <= LAYOUT_PATTERNS['paragraph']['line_spacing']:
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
        
        items.sort(key=lambda x: x['box'][1])
        processed = []
        current_list = []
        
        for item in items:
            if not current_list:
                current_list = [item]
            else:
                last_item = current_list[-1]
                spacing = item['box'][1] - last_item['box'][3]
                
                if spacing <= LAYOUT_PATTERNS['list']['item_spacing']:
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
        
        min_x = min(item['box'][0] for item in items)
        min_y = min(item['box'][1] for item in items)
        max_x = max(item['box'][2] for item in items)
        max_y = max(item['box'][3] for item in items)
        
        max_score = max(item['score'] for item in items)
        labels = [item['label'] for item in items]
        combined_label = ' '.join(labels)
        
        return {
            'box': [min_x, min_y, max_x, max_y],
            'score': max_score,
            'label': combined_label
        }