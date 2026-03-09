import requests

url = "http://localhost:3000/predict"

payload = {
    "property_gfa_total": 50000,
    "year_built": 1995,
    "number_of_floors": 10,
    "primary_property_type": "NonResidential",
    "building_type": "NonResidential",
}

resp = requests.post(url, json=payload)

print("Status:", resp.status_code)
print("Body:", resp.text)
