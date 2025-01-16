# API Endpoints

### 1. Analysis Endpoint (`/api/v1/analyze`)
Analyzes UI elements from uploaded images to extract their characteristics.

#### POST `/api/v1/analyze`
- **URL**: `/api/v1/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

**Request Body**:
- `images`: List of image files (Required)
  - Supported formats: PNG, JPEG
  - Images will be automatically padded to minimum 28x28 dimensions
  - Multiple images can be sent in a single request

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "images=@button1.png" \
  -F "images=@button2.png"
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "type": "button|icon|text|input",
    "text": "exact text if present, null if none",
    "visual_elements": ["icon names or descriptions if present"],
    "primary_function": "main purpose based on visual evidence",
    "dominant_color": "main color if clearly visible, null if unclear"
}
```

**Error Response**:
- **Code**: 500 Internal Server Error
```json
{
    "error": "Error message",
    "type": "ExceptionType"
}
```

### 2. Normalization Endpoint (`/api/v1/normalize`)
Converts natural language UI element descriptions into a standardized format.

#### POST `/api/v1/normalize`
- **URL**: `/api/v1/normalize`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:
```json
{
    "prompt": "Find me a blue button with a phone icon"
}
```

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/normalize" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find me a blue button with a phone icon"}'
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "type": "button",
    "primary_function": "initiate calls/communication",
    "color": "blue",
    "visual_elements": ["phone icon"],
    "derived_intent": "make a phone call"
}
```

**Error Response**:
- **Code**: 500 Internal Server Error
```json
{
    "error": "Error message",
    "type": "ExceptionType",
    "trace": "Detailed error trace"
}
```

### 3. Prefilter Endpoint (`/api/v1/prefilter`)
Pre-screens UI sections to identify potential matches, optimizing the search process.

#### POST `/api/v1/prefilter`
- **URL**: `/api/v1/prefilter`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:
```json
{
    "normalized_prompt": {
        "type": "button",
        "color": "blue",
        "visual_elements": ["phone icon"]
    },
    "sections": [
        {
            "position_metadata": {
                "y_start": 0.0,
                "y_end": 0.5,
                "vertical_position": "top"
            },
            "image": "base64_encoded_image_string"
        }
    ]
}
```

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/prefilter" \
  -H "Content-Type: application/json" \
  -d @prefilter_request.json
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "results": [
        {
            "section_index": 0,
            "position_metadata": {
                "y_start": 0.0,
                "y_end": 0.5,
                "vertical_position": "top"
            },
            "likely_contains": true
        }
    ]
}
```

**Error Response**:
- **Code**: 500 Internal Server Error
```json
{
    "error": "Error message",
    "type": "ExceptionType"
}
```

### 4. Match Endpoint (`/api/v1/match`)
Performs precise matching between normalized prompts and UI elements.

#### POST `/api/v1/match`
- **URL**: `/api/v1/match`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:
```json
{
    "normalized_prompt": {
        "type": "button",
        "color": "blue",
        "visual_elements": ["phone icon"]
    },
    "elements": [
        {
            "id": "button-1",
            "type": "button",
            "visual_elements": ["phone icon"],
            "dominant_color": "blue",
            "text": "Call",
            "primary_function": "initiate call",
            "neighbors": {
                "right": {
                    "id": "text-1",
                    "type": "text",
                    "visual_elements": [],
                    "text": "settings"
                }
            }
        }
    ]
}
```

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/match" \
  -H "Content-Type: application/json" \
  -d @match_request.json
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "match_id": "button-1"
}
```
or if no match found:
```json
{
    "match_id": false
}
```

**Error Response**:
- **Code**: 500 Internal Server Error
```json
{
    "error": "Error message",
    "type": "ExceptionType"
}
```

### 5. Maintenance Endpoints

#### POST `/api/v1/reset`
Resets the LLM and clears CUDA memory.

- **URL**: `/api/v1/reset`
- **Method**: `POST`
- **Content-Type**: `application/json`

**No Request Body Required**

**Example Request using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/reset"
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "status": "success",
    "memory_stats": {
        "before": {
            "ram": {},
            "cuda": {},
            "vllm": {}
        },
        "after": {
            "ram": {},
            "cuda": {},
            "vllm": {}
        }
    }
}
```

#### GET `/api/v1/health`
Returns detailed system health statistics.

- **URL**: `/api/v1/health`
- **Method**: `GET`

**Example Request using curl**:
```bash
curl "http://localhost:8000/api/v1/health"
```

**Success Response**:
- **Code**: 200 OK
```json
{
    "status": "ok",
    "stats": {
        "ram": {
            "total": 32.0,
            "used": 16.5,
            "percent": 51.5
        },
        "cuda": {
            "allocated": 8.5,
            "reserved": 16.0,
            "max_allocated": 12.0,
            "max_reserved": 16.0
        },
        "vllm": {
            "gpu_memory": 16.0,
            "gpu_memory_utilization": 0.85
        }
    }
}
```

**Error Response**:
- **Code**: 503 Service Unavailable
```json
{
    "status": "unavailable",
    "error": "Error message"
}
```

## Configuration

### Environment Settings (settings.py)
```python
MODEL_NAME = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 32768
MAX_NUM_SEQS = 64
WORKERS = 4
HOST = "0.0.0.0"
PORT = 8000
```

## Architecture Components

### 1. LLM Singleton
- Manages a single instance of the Qwen model
- Handles batch processing and memory management
- Provides thread-safe access to the model

### 2. Image Processing
- Supports multiple image formats
- Automatically pads images to minimum dimensions (28x28)
- Converts images to RGB format

### 3. Prompt Templates
- Analysis prompt for detailed UI element inspection
- Normalization prompt for standardizing search queries
- Prefilter prompt for quick section filtering
- Match prompt for precise element comparison

## Error Handling
The service includes comprehensive error handling:
- Invalid image formats
- Memory allocation issues
- JSON parsing errors
- Model inference failures

## Monitoring and Maintenance
- Memory usage statistics for both RAM and CUDA
- Detailed GPU utilization metrics
- Cache management and garbage collection
- System health monitoring

## Best Practices

### Performance Optimization
1. Use batch processing when possible
2. Monitor memory usage through health endpoint
3. Reset the service if memory usage becomes too high
4. Keep image sizes reasonable

### Error Recovery
1. Check health endpoint regularly
2. Use reset endpoint if service becomes unstable
3. Monitor system logs for errors
4. Implement appropriate retry logic in clients

## Common Issues and Solutions

### Memory Issues
- Symptom: Service becomes unresponsive or slow
- Solution: Call the reset endpoint to clear memory

### Image Processing Errors
- Symptom: 500 error on image upload
- Solution: Ensure images are in supported format (PNG, JPEG)
- Solution: Verify image dimensions are reasonable

### Model Inference Timeout
- Symptom: Request timeout on analysis
- Solution: Reduce batch size
- Solution: Split large requests into smaller batches

## Security Considerations
- Service runs on internal port 8000
- No authentication implemented by default
- Consider adding API key authentication for production
- Implement rate limiting for production use

## Development and Testing
1. Use the health endpoint to verify service status
2. Test with various image sizes and formats
3. Monitor memory usage during batch processing
4. Implement appropriate error handling in clients

## Future Improvements
- Add authentication
- Implement request queuing
- Add result caching
- Support for more image formats
- Add detailed logging and monitoring