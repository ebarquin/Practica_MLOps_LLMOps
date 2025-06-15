from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Pipelines de Hugging Face
classifier = pipeline("zero-shot-classification")
translation_pipeline = pipeline("translation_en_to_fr")

@app.get("/")
def root():
    return {"mensaje": "API FastAPI en funcionamiento"}

@app.get("/classify/")
def classify_text(text: str, labels: str):
    label_list = labels.split(",")
    return classifier(text, candidate_labels=label_list)

@app.get("/translate/")
def translate_to_french(text: str):
    return translation_pipeline(text)

@app.get("/palindrome/")
def is_palindrome(word: str):
    return {"palindrome": word == word[::-1]}

@app.get("/reversed/")
def reverse_text(text: str):
    return {"reversed": text[::-1]}

@app.get("/length/")
def get_length(text: str):
    return {"length": len(text)}
