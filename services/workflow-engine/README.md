# Image Processing API Documentation

![alt text](../.assets/image01.png)

## System Overview
This FastAPI-based service processes images through multiple stages of analysis and matching. The system is designed to efficiently handle image processing with caching capabilities and distributed processing across multiple specialized services.

## Service Components

### External Services
- Mask Generation (`mask-generation:8000`)
- Visual Analysis (`qwen2-vl:8000`)
- MongoDB Database (for caching)

## Process Flow Details

### 1. Cache Flow
- **Client Request**: Accepts POST request with image file and prompt
- **Cache Check**: 
  - Generates MD5 hash of incoming image
  - Checks MongoDB for existing results
- **Mask Generation** (on cache miss):
  - Generates image sections
  - Stores results in MongoDB for future use
- **Normalization**:
  - Processes user prompt for standardized matching

### 2. Section Processing
- **Build Map**: Creates a comprehensive map of all sections
- **Filter Process**:
  - Initial filtering of sections against normalized prompt
  - Relaxed retry if no matches found
- **Collection**: Gathers relevant sections for analysis

### 3. Analysis & Matching
- **Batch Processing**: 
  - Processes sections in batches of 100
  - Handles visual analysis tasks
- **Analysis**:
  - Determines element types
  - Extracts visual elements
  - Identifies dominant colors
- **Update & Children**:
  - Updates section map with results
  - Collects child elements
- **Matching Process**:
  - Performs initial matches (5 elements per batch)
  - Reduces multiple matches to find best match

### 4. Response Generation
- **Build**: Constructs base response structure
- **Debug** (optional):
  - Includes additional analysis information
  - Adds section processing details
- **Mask** (optional):
  - Includes complete mask generation results
- **Send**: Returns final processed results

## API Endpoint Details

### POST /process-image
```python
async def process_image(
    file: UploadFile,
    prompt: str,
    include_mask: bool = False,
    debug: bool = False
)
```

#### Parameters
- `file`: Image file (multipart/form-data)
- `prompt`: Text prompt for matching
- `include_mask`: Include mask data in response
- `debug`: Include debug information

#### Response Format
```json
{
    "filtered_section_ids": ["id1", "id2"],
    "children_count": 10,
    "match": {
        "id": "matched_element_id",
        "type": "element_type",
        "visual_elements": []
    },
    "debug": [],  // If debug=true
    "mask_result": {}  // If include_mask=true
}
```
