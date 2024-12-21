import pytest
import requests
import json
import os
from datetime import datetime
from io import BytesIO
import numpy as np
from math import sqrt

class TestRunner:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f'test_run_{self.timestamp}'
        self.metrics = {
            'summary': {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'avg_response_time': 0,
                'avg_distance': 0,
                'avg_iou': 0,
                'avg_confidence': 0,
                'error_types': {}
            },
            'iterations': [],
            'failures': []
        }
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def calculate_iou(self, bbox1, bbox2):
        if not bbox1 or not bbox2:
            return 0.0
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        return intersection / (area1 + area2 - intersection)

    def calculate_distance(self, point1, point2):
        if not point1 or not point2:
            return -1
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _convert_timings(self, timings):
        converted = {}
        for key, value in timings.items():
            if isinstance(value, datetime):
                converted[key] = value.strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                converted[key] = value
        return converted

    def save_failure(self, iteration, image_hex, prompt, bbox, response, error=None, 
                    actual_pos=None, predicted_pos=None):
        failure_data = {
            'iteration': iteration,
            'prompt': prompt,
            'bbox': bbox,
            'response': response,
            'error': str(error) if error else None,
            'actual_pos': actual_pos,
            'predicted_pos': predicted_pos,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        
        if image_hex:
            failure_dir = os.path.join(self.base_dir, f'failure_{iteration}')
            if not os.path.exists(failure_dir):
                os.makedirs(failure_dir)
            with open(os.path.join(failure_dir, 'image.png'), 'wb') as f:
                f.write(bytes.fromhex(image_hex))
                
        self.metrics['failures'].append(failure_data)
        self.save_metrics()

    def add_iteration_metrics(self, iteration, timings, bbox=None, predicted_bbox=None, 
                            success=False, response_data=None, error=None, confidence=None):
        center_point = ([bbox['x'] + bbox['width']/2, bbox['y'] + bbox['height']/2] 
                       if bbox else None)
        pred_point = (predicted_bbox['position'] if predicted_bbox else None)

        metrics = {
            'id': iteration,
            'success': success,
            'timings': self._convert_timings(timings),
            'accuracy': {
                'distance': self.calculate_distance(center_point, pred_point),
                'iou': self.calculate_iou(bbox, predicted_bbox),
                'confidence': confidence or 0
            },
            'response': {
                'status': getattr(response_data, 'status_code', 0),
                'size': len(str(response_data)) if response_data else 0,
                'error': str(error) if error else None
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        
        self.metrics['iterations'].append(metrics)
        self.update_summary(metrics)
        self.save_metrics()

    def update_summary(self, metrics):
        summary = self.metrics['summary']
        summary['total_tests'] += 1
        
        if metrics['success']:
            summary['successful_tests'] += 1
        else:
            summary['failed_tests'] += 1
            error_type = metrics['response']['error'] or 'unknown'
            summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1

        n = summary['total_tests']
        summary['avg_response_time'] = (summary['avg_response_time'] * (n-1) + 
                                      metrics['timings']['total']) / n
        summary['avg_distance'] = (summary['avg_distance'] * (n-1) + 
                                 metrics['accuracy']['distance']) / n
        summary['avg_iou'] = (summary['avg_iou'] * (n-1) + 
                            metrics['accuracy']['iou']) / n
        summary['avg_confidence'] = (summary['avg_confidence'] * (n-1) + 
                                   metrics['accuracy']['confidence']) / n

    def save_metrics(self):
        metrics_file = os.path.join(self.base_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

runner = TestRunner()

@pytest.mark.parametrize('iteration', range(2))
def test_screen(iteration):
    timings = {'start': datetime.now()}
    bbox = None
    result = None
    
    try:
        response = requests.get('http://localhost:5000/random-screen')
        timings['get_screen'] = (datetime.now() - timings['start']).total_seconds()
        
        data = response.json()
        bbox = data['data']['bbox']
        
        proc_response = requests.post(
            'http://localhost:9999/process-image',
            files={'file': ('image.png', BytesIO(bytes.fromhex(data['image'])), 'image/png')},
            data={'prompt': data['data']['prompt']}
        )
        
        if proc_response.status_code == 500:
            pytest.exit("Server returned 500 error")
            
        timings['total'] = (datetime.now() - timings['start']).total_seconds()
        
        result = proc_response.json()
        
        success = False
        predicted_bbox = None
        confidence = None
        
        if result and result.get('match'):
            predicted_pos = result['match']['position']
            predicted_bbox = {
                'x': predicted_pos[0] - 5,
                'y': predicted_pos[1] - 5,
                'width': 10,
                'height': 10,
                'position': predicted_pos
            }
            confidence = result['match'].get('confidence', 0)
            success = (bbox['x'] <= predicted_pos[0] <= bbox['x'] + bbox['width'] and
                      bbox['y'] <= predicted_pos[1] <= bbox['y'] + bbox['height'])
        
        runner.add_iteration_metrics(
            iteration=iteration,
            timings=timings,
            bbox=bbox,
            predicted_bbox=predicted_bbox,
            success=success,
            response_data=result,
            confidence=confidence
        )
        
        if not success:
            center = [bbox['x'] + bbox['width']/2, bbox['y'] + bbox['height']/2]
            runner.save_failure(
                iteration,
                data['image'],
                data['data']['prompt'],
                bbox,
                result,
                actual_pos=center,
                predicted_pos=predicted_bbox['position'] if predicted_bbox else None
            )
            pytest.fail(f"Test failed: iteration {iteration}")
            
    except Exception as e:
        timings['total'] = (datetime.now() - timings['start']).total_seconds()
        runner.add_iteration_metrics(
            iteration=iteration,
            timings=timings,
            bbox=bbox,
            predicted_bbox=None,
            success=False,
            response_data=result,
            error=e
        )
        pytest.fail(f"Error: {str(e)}")

def pytest_sessionfinish(session):
    runner.save_metrics()