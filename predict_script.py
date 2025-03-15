from transformers import BertForSequenceClassification, BertTokenizer
import torch
from googletrans import Translator, LANGUAGES

# Charger le modèle
model = BertForSequenceClassification.from_pretrained('./clinical_biobert_finetuned', num_labels=8)
tokenizer = BertTokenizer.from_pretrained("./clinical_biobert_finetuned")
translator = Translator()

# Predictions fnction
def predict_birads(report_text):
    # Traduction avant la prédiction
    detected_lang = translator.detect(report_text).lang.split('-')[0]
    if detected_lang != 'en':
        report_text = translator.translate(report_text, src=detected_lang, dest='en').text

    print("Translated...")

    # Tokenize the input report
    inputs = tokenizer(report_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label (index with the highest score)
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
Doente observada a 2010 01.

Os estudos mamográfico e ecográfico documentam área nodular localizada na transição dos
quadrantes internos da mama esquerda com 15 mm de diâmetro, de achados suspeitos de
malignidade.
Efectuou-se microbiopsia ecoguiada onde foram colhidos 4 fragmentos, para estudo anatomopatológico.
Axila positiva. Efectuou-se biopsia aspirativa glanglio-axilar.

Achados imagiológicos muito sugestivos de malignidade - Bi-Rads - 5.
"""

predicted_birads = predict_birads(report)

print(f"Predicted BI-RADS level: {predicted_birads}")
