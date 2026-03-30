import requests
import os
def test():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={os.environ['GEMINI_API_KEY']}"
    response = requests.get(url)
    models = response.json().get('models', [])
    for m in models:
        if "gemini" in m['name']:
            print(m['name'], m.get('supportedGenerationMethods'))
test()
