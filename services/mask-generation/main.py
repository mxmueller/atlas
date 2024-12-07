from src.detector import RefinedUIDetector

def main():
    detector = RefinedUIDetector()
    
    image_path = "./test/screenshots/home.png"  # Passe den Pfad zu deinem Bild an
    output_path = "detected_ui_elements_refined.png"
    
    print(f"Starting detection on {image_path}")
    detector.process_and_visualize(image_path, output_path)

if __name__ == "__main__":
    main()