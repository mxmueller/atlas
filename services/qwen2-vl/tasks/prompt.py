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
       "Extract UI properties. Focus on position details.\n"
       "<|im_end|>\n"
       "<|im_start|>user\n"
       "{{\"type\": \"element type\","
       "\"text\": \"text if present\","
       "\"visual\": [\"icons\"],"
       "\"color\": \"color if specified\","
       "\"position\": \"exact position\"}}\n"
       "Normalize: I need a blue home button with a house icon in the top left corner\n"
       "<|im_end|>\n"
       "<|im_start|>assistant\n"
       "{{\"type\": \"button\","
       "\"text\": \"home\","
       "\"visual\": [\"house icon\"],"
       "\"color\": \"blue\","
       "\"position\": \"top left\"}}\n"
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