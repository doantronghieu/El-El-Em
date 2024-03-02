import requests

if __name__ == "__main__":
    while True:
        content = input('>> ')

        # Make a POST request to the API endpoint
        # Change to your API URL if necessary
        api_url = "http://localhost:8000/chatbot/"
        payload = {"content": content}

        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            api_response = response.json()
            print(f"=> {api_response['response']}")
        else:
            print("Error:", response.status_code)

# python main.py