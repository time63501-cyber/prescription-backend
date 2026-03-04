import google.generativeai as genai
genai.configure(api_key="AIzaSyDaFNKCzgR4juourERszyrS58ISYYMvHYk")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)