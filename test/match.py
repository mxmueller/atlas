import asyncio
import httpx

async def test_match():
   url = "http://localhost:8000/api/v1/match"
   
   test_element1 = {
       "id": "btn1",
       "type": "button",
       "text": "Cancel",
       "visual_elements": ["x-icon"],
       "dominant_color": "red",
       "primary_function": "This icon likely represents to abort or cancle.",
       "neighbors": {
           "left": {
               "id": "btn2",
               "type": "button", 
               "text": "Save",
               "visual_elements": ["save-icon"],
               "dominant_color": "green"
           }
       }
   }
   
   test_element2 = {
       "id": "btn2",
       "type": "button",
       "text": "Text", 
       "visual_elements": ["Text-icon"],
       "dominant_color": "blue",
       "primary_function": "This icon likely represents a text or document related function, possibly for creating or editing documents.",
        "neighbors": {
           "right": {
               "id": "btn3",
               "type": "button", 
               "text": "Home",
               "visual_elements": ["home-icon"],
               "dominant_color": "yellow"
           }
       }
   }

   test_data = {
       "normalized_prompt": {
           "type": "button",
           "text": "Print the Document", 
           "color": "blue",
           "position": "top right"
       },
       "elements": [test_element1, test_element2]
   }

   async with httpx.AsyncClient() as client:
       response = await client.post(url, json=test_data)
       print(f"Status Code: {response.status_code}")
       print(f"Response: {response.json()}")

if __name__ == "__main__":
   asyncio.run(test_match())