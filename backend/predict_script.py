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
Patient treated in 2009 02. The imaging study documents a nodule located in the left breast QSE with a diameter of 6 cm.
An ultrasound-guided microbiopsy was performed, in which 4 fragments were collected for anatomopathological study.
A lymph node axillary aspiration biopsy was performed.
Preoperative carbon marking was also performed.
"""

predicted_birads = predict_birads(report)

print(f"Predicted BI-RADS level: {predicted_birads}")
