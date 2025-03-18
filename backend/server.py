from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification


app = FastAPI()

# Load model and tokenizer once
model_path = './clinical_biobert_finetuned'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)


class TextRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(request: TextRequest):
    # Tokenize the input report
    inputs = tokenizer(request.report, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label (index with highest score)
    predicted_class = torch.argmax(logits, dim=-1).item()

    birads_mapping = {
        0: "2",
        1: "3",
        2: "4",
        3: "4a",
        4: "4b",
        5: "4c",
        6: "5",
        7: "6"
    }
    prediction = birads_mapping.get(predicted_class, 'Unknown')  
    # Return the corresponding BI-RADS value
    return { "prediction" : prediction }

































# Define input format
class TextRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(request: TextRequest):
    # Tokenize input text
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return {"prediction": predicted_label}

