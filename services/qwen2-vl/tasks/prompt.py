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
        "You are an extremely lenient UI section analyzer. Your goal is to INCLUDE as many potential matches as possible.\n"
        "IMPORTANT: When in doubt, ALWAYS return true. Only return false if you are 100% certain there is NO possibility of a match.\n"
        "Matching rules:\n"
        "- ANY interactive element could be a potential match\n"
        "- ANY color should be considered similar enough\n"
        "- Positions are just rough guidelines\n"
        "- Text matching should allow for variations\n"
        "- Missing attributes should be ignored completely\n"
        "- Different shapes/sizes are acceptable\n"
        "- Different visual styles are acceptable\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Find ANY UI elements that could potentially match this description: {str(normalized_prompt)}\n"
        "Remember: When unsure, include it!\n"
        "<|vision_start|>"
        "<|image_pad|>"
        "<|vision_end|>\n"
        "Could this section contain anything similar? Return JSON:\n"
        "{\n"
        '    "contains": true|false\n'
        "}\n"
        "REMEMBER: Default to true unless absolutely certain!\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return base_prompt