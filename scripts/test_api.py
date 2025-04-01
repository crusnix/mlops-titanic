import requests

url = "http://127.0.0.1:8000/predict/"
data = {"Pclass": 3, "Age": 25, "Fare": 7.25}  # Sample input

response = requests.post(url, json=data)  # Send request
print("Response:", response.json())  # Print response
