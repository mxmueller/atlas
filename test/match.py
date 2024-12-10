import asyncio
import httpx

async def test_match():
    url = "http://localhost:8000/api/v1/match"
    
    # Test data
    test_element1 = {
        "id": "btn1",
        "type": "button",
        "text": "Cancel",
        "visual_elements": ["x-icon"],
        "primary_function": "Cancels current operation",
        "dominant_color": "red",
        "neighbours": {
            "left": {
                "id": "btn2",
                "type": "button",
                "text": "Save",
                "visual_elements": ["save-icon"],
                "primary_function": "Saves current state",
                "dominant_color": "green",
                "neighbours": {}
            },
            "right": None,
            "above": None,
            "below": None
        }
    }
    
    test_element2 = {
        "id": "btn3",
        "type": "button",
        "text": "Cancel",
        "visual_elements": ["x-icon"],
        "primary_function": "Cancels current operation",
        "dominant_color": "red",
        "neighbours": {
            "left": None,
            "right": None,
            "above": None,
            "below": None
        }
    }

    # Test request
    test_data = {
        "normalized_prompt": {
            "type": "button",
            "text": "cancel",
            "color": "red",
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