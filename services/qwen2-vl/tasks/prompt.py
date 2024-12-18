def create_analysis_prompt() -> str:
    base_prompt = (
        "<|im_start|>system\n"
        "You are a precise UI element analyzer. Extract information ONLY from what you can see.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Example output format:\n"
        "{\n"
        '    "type": "button|icon|text|input",\n'
        '    "text": "exact text if present, null if none",\n'
        '    "visual_elements": ["icon names or descriptions if present else put none"],\n'
        '    "primary_function": "main purpose based on visual evidence only make two sentence",\n'
        '    "dominant_color": "main color if clearly visible, null if unclear"\n'
        "}\n\n"
    )
    return (
        f"{base_prompt}"
        "<|vision_start|>"
        "<|image_pad|>"
        "<|vision_end|>\n"
        "Analyze this UI element and return valid JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def create_normalization_prompt() -> str:
    template = (
        "<|im_start|>system\n"
        "Extract UI properties with focus on function and context.\n"
        "For neighbors use ONLY these directions: \"left\", \"right\", \"above\", \"below\", \"undefined\". Other positions are invalid.\n"
        "Always include a brief functional summary based on all available information.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Find me a blue phone button with phone icon\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{{\n"
        "  \"type\": \"button\",\n"
        "  \"primary_function\": \"initiate calls/communication\",\n"
        "  \"color\": \"blue\",\n"
        "  \"visual_elements\": [\"phone icon\"],\n"
        "  \"derived_intent\": \"make a phone call\"\n"
        "}}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Find me a green phone button with text call next to settings\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{{\n"
        "  \"type\": \"button\",\n"
        "  \"text\": \"call\",\n"
        "  \"primary_function\": \"initiate calls/communication\",\n"
        "  \"color\": \"green\",\n"
        "  \"neighbors\": {{\n"
        "    \"right\": {{\n"
        "      \"type\": \"text\",\n"
        "      \"text\": \"settings\"\n"
        "    }}\n"
        "  }},\n"
        "  \"derived_intent\": \"access phone functionality near settings\"\n"
        "}}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Normalize: {0}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return template

def create_prefilter_prompt(normalized_prompt: dict) -> str:
    base_prompt = (
        "<|im_start|>system\n"
        "You are a UI section analyzer. Determine if a UI element could exist in this section.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Looking for this UI element: {str(normalized_prompt)}\n"
        "<|vision_start|>"
        "<|image_pad|>"
        "<|vision_end|>\n"
        "Could this section contain the described element? Return only valid JSON:\n"
        "{\n"
        '    "contains": true|false\n'
        "}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return base_prompt
