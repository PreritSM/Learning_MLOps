import requests
import json
url = "http://localhost:5000/predict"
data = {
"CHAS": {"0": 0},
"RM": {"0": 6.575},
"TAX": {"0": 296.0},
"PTRATIO": {"0": 15.3},
"B": {"0": 396.9},
"LSTAT": {"0": 4.98},
}
# Convert to JSON string
input_data = json.dumps(data)
# Set the content type
headers = {"Content-Type": "application/json"}
# Make the request and display the response
resp = requests.post(url, input_data, headers=headers)
print(resp.text)