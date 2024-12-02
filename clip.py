#!/usr/bin/env python3
"""
UI Button Analyzer - Ein Tool zur Analyse von UI-Buttons mittels LLaVA Modell und OCR.
Extrahiert visuelle und funktionale Eigenschaften sowie Text in Deutsch und Englisch.
"""

import argparse
from typing import Dict, Any, List, Tuple
from PIL import Image
from vllm import LLM, SamplingParams
import easyocr
import numpy as np


def create_llm() -> LLM:
    """
    Erstellt und konfiguriert eine LLM-Instanz mit optimierten Parametern.
    
    Returns:
        LLM: Konfigurierte LLM-Instanz
    """
    return LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=2048,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.95,
        max_num_seqs=1,
        enforce_eager=True
    )


def create_ocr_readers() -> Tuple[easyocr.Reader, easyocr.Reader]:
    """
    Erstellt OCR-Reader für Deutsch und Englisch.
    
    Returns:
        Tuple[easyocr.Reader, easyocr.Reader]: Tuple mit deutschen und englischen OCR-Readern
    """
    # Initialisiere Reader für deutsche Texte
    german_reader = easyocr.Reader(['de'], gpu=True)
    # Initialisiere Reader für englische Texte
    english_reader = easyocr.Reader(['en'], gpu=True)
    
    return german_reader, english_reader


def extract_text(image_path: str, readers: Tuple[easyocr.Reader, easyocr.Reader]) -> Dict[str, List[str]]:
    """
    Extrahiert Text aus dem Bild in beiden Sprachen.
    
    Args:
        image_path (str): Pfad zum Bild
        readers (Tuple[easyocr.Reader, easyocr.Reader]): OCR-Reader für beide Sprachen
    
    Returns:
        Dict[str, List[str]]: Extrahierte Texte in beiden Sprachen
    """
    german_reader, english_reader = readers
    
    # Lade das Bild
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # Extrahiere Text in beiden Sprachen
    german_results = german_reader.readtext(image_np)
    english_results = english_reader.readtext(image_np)
    
    # Extrahiere nur den Text aus den Ergebnissen
    german_texts = [result[1] for result in german_results]
    english_texts = [result[1] for result in english_results]
    
    return {
        "german_text": german_texts,
        "english_text": english_texts
    }


def get_analysis_prompt() -> str:
    """
    Generiert einen präzisen Prompt für die Button-Analyse.
    
    Returns:
        str: Formatierter Prompt-String
    """
    return """USER: <image> Analyze this UI button and provide detailed JSON with:
1. Element type and style
2. Colors (background, icon)
3. Icon details
4. Function and context
Format as JSON only:
{
    "element": {"type": "", "style": ""},
    "colors": {"background": "", "icon": ""},
    "icon": {"type": "", "design": ""},
    "function": {"purpose": "", "context": ""}
}
ASSISTANT:"""


def get_sampling_params() -> SamplingParams:
    """
    Erstellt optimierte Sampling-Parameter für die Modell-Inferenz.
    
    Returns:
        SamplingParams: Konfigurierte Sampling-Parameter
    """
    return SamplingParams(
        temperature=0.1,
        max_tokens=256,
        top_p=0.95,
        presence_penalty=0.1
    )


def analyze_button(image_path: str, model_type: str = "llava") -> Dict[str, Any]:
    """
    Analysiert einen UI-Button und extrahiert dessen Eigenschaften und Text.
    
    Args:
        image_path (str): Pfad zum Button-Bild
        model_type (str, optional): Zu verwendendes Modell. Defaults to "llava".
    
    Returns:
        Dict[str, Any]: Dictionary mit Analyse-Ergebnissen und extrahiertem Text
        
    Raises:
        ValueError: Bei ungültigem Modelltyp
        FileNotFoundError: Wenn das Bild nicht gefunden wird
    """
    if model_type != "llava":
        raise ValueError(f"Nicht unterstützter Modelltyp: {model_type}")
        
    image = Image.open(image_path).convert('RGB')
    
    # LLaVA Analyse
    llm = create_llm()
    sampling_params = get_sampling_params()
    
    inputs = {
        "prompt": get_analysis_prompt(),
        "multi_modal_data": {
            "image": image
        },
    }
    
    visual_analysis = llm.generate(inputs, sampling_params=sampling_params)[0].outputs[0].text
    
    # Textextraktion
    readers = create_ocr_readers()
    extracted_text = extract_text(image_path, readers)
    
    # Kombiniere die Ergebnisse
    results = {
        "visual_analysis": visual_analysis,
        "text_content": extracted_text
    }
    
    return results


def parse_arguments() -> argparse.Namespace:
    """
    Verarbeitet Kommandozeilenargumente.
    
    Returns:
        argparse.Namespace: Parsierte Argumente
    """
    parser = argparse.ArgumentParser(
        description='Analysiert UI-Buttons mithilfe von LLaVA und extrahiert Text'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Pfad zum Button-Bild'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default="llava",
        choices=["llava"],
        help='Zu verwendendes Modell'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Hauptfunktion des Programms.
    Verarbeitet Argumente und führt die Button-Analyse durch.
    """
    args = parse_arguments()
    
    try:
        results = analyze_button(args.image_path, args.model_type)
        print("Visuelle Analyse:")
        print(results["visual_analysis"])
        print("\nExtrahierter Text:")
        print("Deutsch:", results["text_content"]["german_text"])
        print("Englisch:", results["text_content"]["english_text"])
    except Exception as e:
        print(f"Fehler bei der Button-Analyse: {str(e)}")
        raise


if __name__ == "__main__":
    main()