from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

label_map = {0: 'Positif', 1: 'Netral', 2: 'Negatif'}

def analyze_sentiment(df):

    def get_sentiment(text):
        if pd.isna(text):
            return 'Netral'

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

        return label_map[pred]

    df['sentiment'] = df['content'].apply(get_sentiment)

    # simpan ke processed
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/hasil_sentimen.csv', index=False)

    print("Analisis sentimen selesai")

    return df
