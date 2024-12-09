MODEL_CONFIG = {
    'model_id': "IDEA-Research/grounding-dino-base",
}

TEXT_DETECTION_PARAMS = {
    'min_text_length': 1,        # Reduziert von 2 auf 1 für einzelne Buchstaben/Zahlen
    'max_text_gap': 50,          # Erhöht von 30 auf 50 für mehr Flexibilität bei Textabständen
    'line_height_tolerance': 12,  # Erhöht von 8 auf 12 für verschiedene Schriftgrößen
    'word_spacing': 25,          # Erhöht von 15 auf 25 für großzügigere Wortabstände
    'menu_item_max_gap': 60,     # Erhöht von 40 auf 60 für Menüeinträge mit größeren Abständen
    'paragraph_line_spacing': 8,  # Erhöht von 5 auf 8
    'list_item_indent': 20,      # Bleibt gleich
    'heading_min_height': 20     # Reduziert von 25 auf 20 für kleinere Überschriften
}

LAYOUT_PATTERNS = {
    'menu_bar': {
        'height_range': (20, 40),
        'min_items': 3,
        'max_gap': 40
    },
    'paragraph': {
        'min_width': 200,
        'min_lines': 2,
        'line_spacing': 5
    },
    'list': {
        'indent': 20,
        'bullet_width': 15,
        'item_spacing': 8
    }
}

PROMPTS = [
    "a single button. a standalone button. an isolated button."
    "a icon. a house icon. a settings gear icon. a menu hamburger icon. a search magnifying glass icon. a user profile icon. a home icon.",
    "a close x icon. a plus add icon. a minus remove icon. a download icon. a share icon. an edit pencil icon.",
    "a facebook icon. a twitter bird icon. an instagram camera icon. a linkedin icon. a youtube play icon.",
    "a notification bell icon. an info icon. a check success icon. an error icon.",
    "a play triangle icon. a pause icon. a stop square icon. a next icon. a previous icon. a volume speaker icon.",
    "a document file icon. a pdf icon. an image photo icon. a folder icon. an attachment clip icon.",
]

VISUALIZATION_COLORS = {
    'menu': (52, 152, 219),
    'text': (46, 204, 113),
    'list': (155, 89, 182),
    'button': (231, 76, 60),
    'label': (241, 196, 15),
    'default': (149, 165, 166)
}