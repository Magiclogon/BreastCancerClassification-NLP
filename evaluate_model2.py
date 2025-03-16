import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from Preprocessing.avoiding_biased_data import balanced_df

# Load the tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset (assuming balanced_df is already defined)
df = balanced_df
df = df[['translated_text', 'BI-RADS']]

# Convert BI-RADS categories to numeric values
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

# Remove unknown values
df = df[df['BI-RADS'] != 'Unknown']

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Tokenization function
def tokenize_and_format_function(examples):
    tokenized = tokenizer(examples['translated_text'], padding='max_length', truncation=True, max_length=512)
    tokenized['labels'] = [label - 2 for label in examples['BI-RADS']]  # Adjust labels to start from 0
    return tokenized

# Tokenize the entire dataset
tokenized_dataset = dataset.map(tokenize_and_format_function, batched=True)

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('./modern_bert_finetuned', num_labels=8)

# Define metrics calculation function
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define evaluation arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=16,
    no_cuda=False,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,  # Use the entire dataset for evaluation
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model
print("Evaluating model on the entire dataset...")
evaluation_results = trainer.evaluate()

# Display evaluation results
print("Evaluation results:", evaluation_results)

# Optional: Obtain predictions for further analysis
predictions = trainer.predict(tokenized_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Create a confusion matrix (optional)
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

true_labels = predictions.label_ids

# Display detailed classification report
class_names = ["BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 4A",
               "BI-RADS 4B", "BI-RADS 4C", "BI-RADS 5", "BI-RADS 6"]
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("\nClassification Report:")
print(report)

# Save results to file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Accuracy: {evaluation_results['eval_accuracy']}\n")
    f.write("\nConfusion Matrix:\n")
    f.write("\n\nClassification Report:\n")
    f.write(report)

print("Evaluation complete. Results saved to 'evaluation_results.txt'")