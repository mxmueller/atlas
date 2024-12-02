import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import argparse

class ButtonAnalyzer:
    def __init__(self):
        try:
            # Optimierungen für T4
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16,  # float16 statt bfloat16 für bessere T4-Kompatibilität
                device_map="auto"
            )
            
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct"
            )
            print("Model initialized successfully")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def analyze_button(self, image_path, output_file=None):
        try:
            if not image_path.startswith(('http://', 'https://', 'data:')):
                image_path = f"file://{image_path}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": """Analyze this button and provide in JSON:
                        - text content if visible
                        - main colors (background and text)
                        - icon description if present
                        - likely purpose in one short phrase"""}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            with torch.cuda.amp.autocast():
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = inputs.to("cuda")
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            try:
                json_str = output_text.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:-3]
                result = json.loads(json_str)
                
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to {output_file}")
                
                return result
                
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse response",
                    "raw_response": output_text
                }
                
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Analyze button screenshots')
    parser.add_argument('image_path', help='Path to button screenshot')
    parser.add_argument('--output', '-o', help='Output JSON file path', default=None)
    args = parser.parse_args()

    try:
        analyzer = ButtonAnalyzer()
        result = analyzer.analyze_button(args.image_path, args.output)
        if not args.output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()