from transformers import BertForSequenceClassification, BertTokenizer
import torch
import sys

# Charger le modèle
model = BertForSequenceClassification.from_pretrained('./clinical_biobert_finetuned', num_labels=8)
tokenizer = BertTokenizer.from_pretrained("./clinical_biobert_finetuned")


# Predictions fnction
def predict_birads(report_text):
    # Tokenize the input report
    inputs = tokenizer(report_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
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

    # Return the corresponding BI-RADS value
    return birads_mapping.get(predicted_class, 'Unknown')

report = """
Doente observada a 2010 01 com antecedentes de mastectomia direita.
O estudo mamográfico efectuado à mama esquerda, não são aparentes imagens de opacidades
nodulares que sugiram malignidade, agrupamentos de microcalcificações suspeitas, nem
outros sinais sugestivos de processo neoplásico ou recidiva.

Achados imagiológicos benignos - Bi-Rads - 2.
"""

predicted_birads = predict_birads(report)

print(f"Predicted BI-RADS level: {predicted_birads}")
