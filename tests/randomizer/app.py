from flask import Flask, render_template, jsonify
import random
from data import COLORS, BUTTON_LABELS, ICONS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

app = Flask(__name__)
CURRENT_BUTTONS = None

def hex_to_rgb(hex_color):
  hex_color = hex_color.lstrip('#')
  return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def is_light_color(hex_color):
  r, g, b = hex_to_rgb(hex_color)
  brightness = (r * 299 + g * 587 + b * 114) / 1000
  return brightness > 128

def get_random_color():
  color_hex = random.choice(list(COLORS.keys()))
  return {
      'hex': color_hex,
      'name': COLORS[color_hex],
      'is_light': is_light_color(color_hex)
  }

def get_button_content():
  display_type = random.choice(['text', 'icon', 'both'])
  return {
      'type': display_type,
      'label': random.choice(BUTTON_LABELS),
      'icon': random.choice(ICONS) 
  }

def generate_buttons():
   buttons = []
   for i in range(9):
       color = get_random_color()
       content = get_button_content()
       buttons.append({
           'id': i + 1,
           'color': color['hex'],
           'color_name': color['name'],
           'is_light': color['is_light'],
           'display_type': content['type'],
           'label': content['label'],
           'icon': content['icon']
       })
   return buttons

def get_button_position(button1, button2):
    """Return relative position of button2 to button1"""
    id1 = button1['id'] - 1  # Convert to 0-based index
    id2 = button2['id'] - 1
    
    # In 3x3 grid
    row1, col1 = id1 // 3, id1 % 3
    row2, col2 = id2 // 3, id2 % 3
    
    if row2 == row1 - 1 and col2 == col1:
        return "below"  # Geändert von "above"
    if row2 == row1 + 1 and col2 == col1:
        return "above"  # Geändert von "below"
    if row2 == row1 and col2 == col1 - 1:
        return "to the right of"  # Geändert von "to the left of"
    if row2 == row1 and col2 == col1 + 1:
        return "to the left of"  # Geändert von "to the right of"
    return None

def generate_prompt(selected_button, all_buttons):
    starters = [
        "I'm looking for",
        "Can you find",
        "Please locate", 
        "I need to find",
        "Could you help me find",
        "Try to find",
        "I want to click on",
        "Show me"
    ]
    
    color_prefix = "the" if random.random() > 0.5 else "a"
    base = f"{random.choice(starters)} {color_prefix} {selected_button['color_name']}"
    
    if selected_button['display_type'] == 'text':
        text_formats = [
            f"button labeled {selected_button['label']}",
            f"button that says {selected_button['label']}",
            f"{selected_button['label']} button"
        ]
        base += f" {random.choice(text_formats)}"
    elif selected_button['display_type'] == 'both':
        icon_name = selected_button['icon'].replace('-', ' ')
        both_formats = [
            f"button showing {selected_button['label']} with {icon_name} icon",
            f"{selected_button['label']} button with {icon_name} icon",
            f"button with {selected_button['label']} text and {icon_name} symbol"
        ]
        base += f" {random.choice(both_formats)}"
    else:
        icon_name = selected_button['icon'].replace('-', ' ')
        icon_formats = [
            f"button with {icon_name} icon",
            f"button showing only {icon_name} symbol",
            f"{icon_name} button"
        ]
        base += f" {random.choice(icon_formats)}"

    # Find buttons that are actually in valid positions relative to selected button
    valid_context_buttons = []
    for other_button in all_buttons:
        if other_button['id'] != selected_button['id']:
            pos = get_button_position(selected_button, other_button)
            if pos:
                valid_context_buttons.append((other_button, pos))
    
    if valid_context_buttons and random.random() > 0.4:
        context_button, pos = random.choice(valid_context_buttons)
        
        position_variants = {
            "above": ["just above", "directly above", "right above"],
            "below": ["just below", "underneath", "right under"],
            "to the left of": ["to the left of", "on the left side of"],
            "to the right of": ["to the right of", "on the right side of"]
        }
        
        pos_desc = random.choice(position_variants[pos])
        context = f" {random.choice(['which is', 'it is', 'you can find it'])} {pos_desc} "
        context += f"the {context_button['color_name']}"
        if context_button['display_type'] in ['text', 'both']:
            context += f" {context_button['label']} button"
        else:
            context += " button"
        base += context

    return base

def setup_driver():
  chrome_options = Options()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  chrome_options.binary_location = '/usr/bin/chromium'
  return webdriver.Chrome(options=chrome_options)

@app.route('/')
def index():
   global CURRENT_BUTTONS
   buttons = CURRENT_BUTTONS if CURRENT_BUTTONS else generate_buttons()
   return render_template('index.html', buttons=buttons)

@app.route('/random-screen')
def random_screen():
   global CURRENT_BUTTONS 
   CURRENT_BUTTONS = generate_buttons()
   
   driver = setup_driver()
   driver.get('http://localhost:5000')
   driver.set_window_size(800, 800)
   
   selenium_buttons = driver.find_elements(By.CLASS_NAME, "grid-button")
   selected_button = random.choice(CURRENT_BUTTONS)
   target_button = selenium_buttons[selected_button['id'] - 1]
   
   location = target_button.location
   size = target_button.size
   
   bbox = {
       "x": location['x'],
       "y": location['y'],
       "width": size['width'],
       "height": size['height']
   }
   
   img = driver.get_screenshot_as_png()
   driver.quit()
   
   prompt = generate_prompt(selected_button, CURRENT_BUTTONS)
   
   return jsonify({
       "image": img.hex(),
       "data": {
           "bbox": bbox,
           "prompt": prompt
       }
   })

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')