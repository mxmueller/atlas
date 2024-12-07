from typing import List, Dict
from config.settings import TEXT_DETECTION_PARAMS, LAYOUT_PATTERNS

class LayoutProcessor:
    def process_layout(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
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