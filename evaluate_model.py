import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer

model_name = "dmis-lab/biobert-v1.1"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load your dataset (CSV with 'translated_text' and 'BI-RADS' columns)
df = pd.read_csv("C:\\Users\\Walid\\Documents\\EMI\\SEMESTRE 3\\Projet-Traitement-Texte\\dataset.csv")

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

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

# Split dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.2)
test_dataset = train_test_split['test']

# Tokenization function
def tokenize_and_format_function(examples):
    tokenized = tokenizer(examples['translated_text'], padding='max_length', truncation=True, max_length=512)
    tokenized['labels'] = [label - 2 for label in examples['BI-RADS']]  # Adjust labels to start from 0
    return tokenized

# Tokenize the test dataset
test_dataset = test_dataset.map(tokenize_and_format_function, batched=True)

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('./biobert_finetuned', num_labels=8)

# Define the evaluation arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    per_device_eval_batch_size=16,   # Batch size for evaluation
    no_cuda=False,                   # Set to True if you don't want to use GPU
)

# Define the Trainer
trainer = Trainer(
    model=model,                         # The trained model to evaluate
    args=training_args,                  # Evaluation arguments
    eval_dataset=test_dataset,           # Test dataset
    tokenizer=tokenizer,                 # Tokenizer for encoding input text
)

# Evaluate the model
print("Evaluation started")
evaluation_results = trainer.evaluate()

# Print the evaluation results
print("Evaluation results:", evaluation_results)
