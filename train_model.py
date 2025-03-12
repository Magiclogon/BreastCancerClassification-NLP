import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from Preprocessing.avoiding_biased_data import balanced_df
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)

df = balanced_df

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

# To huggingface format
dataset = Dataset.from_pandas(df)

# Diviser en données de tests et d'entrainement
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print("Processing done!")

# FINETUNING
def tokenize_and_format_function(examples):
    tokenized = tokenizer(examples['translated_text'], padding='max_length', truncation=True, max_length=512)
    tokenized['labels'] = [label - 2 for label in examples['BI-RADS']]
    return tokenized

train_dataset = train_dataset.map(tokenize_and_format_function, batched=True)
test_dataset = test_dataset.map(tokenize_and_format_function, batched=True)

# Charger le modèle
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8).to(device)
print("Modèle chargé")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    evaluation_strategy="epoch",     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for optimization
    save_steps=10_000,               # Save model every 10,000 steps
    save_total_limit=2,              # Limit the total number of saved models
)

# Define the Trainer
trainer = Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=test_dataset,           # Evaluation dataset
    tokenizer=tokenizer,                 # Tokenizer
)

print("Entrainement commencé")
# Train the model
trainer.train()

# Save the model
model.save_pretrained('./clinical_biobert_finetuned')
tokenizer.save_pretrained('./clinical_biobert_finetuned')

