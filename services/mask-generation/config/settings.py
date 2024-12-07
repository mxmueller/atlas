MODEL_CONFIG = {
    'model_id': "IDEA-Research/grounding-dino-base",
}

TEXT_DETECTION_PARAMS = {
    'min_text_length': 2,
    'max_text_gap': 30,
    'line_height_tolerance': 8,
    'word_spacing': 15,
    'menu_item_max_gap': 40,
    'paragraph_line_spacing': 5,
    'list_item_indent': 20,
    'heading_min_height': 25
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
    "a horizontal menu item text. a navigation bar text. a top menu text. a menu item label. a menu bar item.",
    "a paragraph text. a continuous text block. a multi-line text. a text content block. an article text. a description text.",
    "a bullet point text. a list item text. a numbered item text. an indented text. a list entry text.",
    "a button text. a link text. a clickable text. an action text. a menu button text.",
    "a heading text. a title text. a label text. a caption text. a section text."
]

VISUALIZATION_COLORS = {
    'menu': (52, 152, 219),
    'text': (46, 204, 113),
    'list': (155, 89, 182),
    'button': (231, 76, 60),
    'label': (241, 196, 15),
    'default': (149, 165, 166)
}