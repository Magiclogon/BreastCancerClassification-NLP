import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer

# Charger le tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Charger le dataset (CSV avec 'translated_text' et 'BI-RADS')
df = pd.read_csv("dataset.csv")

df = df[['translated_text', 'BI-RADS']]
df['BI-RADS'] = df['BI-RADS'].apply(
    lambda x: (
        2 if x == '2' else
        3 if x == '3' else
        4 if x == '4' else
        5 if x == '4A' or x == '4a' else
        6 if x == '4B' or x == '4b' else
        7 if x == '4C' or x == '4c' else
        8 if x == '5' else
        9 if x == '6' else
        'Unknown'
    )
)

df = df[df['BI-RADS'] != 'Unknown']  # Supprimer les valeurs inconnues

# Convertir en dataset Hugging Face
dataset = Dataset.from_pandas(df)

# Diviser en train et test
train_test_split = dataset.train_test_split(test_size=0.2)
test_dataset = train_test_split['test']

# Fonction de tokenization
def tokenize_and_format_function(examples):
    tokenized = tokenizer(examples['translated_text'], padding='max_length', truncation=True, max_length=512)
    tokenized['labels'] = [label - 2 for label in examples['BI-RADS']]  # Ajuster les labels pour commencer à 0
    return tokenized

# Tokeniser le dataset de test
test_dataset = test_dataset.map(tokenize_and_format_function, batched=True)

# Charger le modèle fine-tuné
model = BertForSequenceClassification.from_pretrained('./biobert_finetuned', num_labels=8)

# Définir la fonction de calcul de l'accuracy
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Définir les arguments d'évaluation
training_args = TrainingArguments(
    output_dir='./results',          # Répertoire de sortie
    per_device_eval_batch_size=16,   # Taille du batch pour l'évaluation
    no_cuda=False,                   # Utiliser GPU si disponible
)

# Définir le Trainer
trainer = Trainer(
    model=model,                         # Modèle entraîné à évaluer
    args=training_args,                  # Arguments d'évaluation
    eval_dataset=test_dataset,           # Dataset de test
    tokenizer=tokenizer,                 # Tokenizer pour encoder les textes
    compute_metrics=compute_metrics,     # Ajout du calcul d'accuracy
)

# Évaluer le modèle
print("Évaluation en cours...")
evaluation_results = trainer.evaluate()

# Afficher les résultats de l'évaluation
print("Résultats de l'évaluation:", evaluation_results)
