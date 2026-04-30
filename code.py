!pip install google-play-scraper vaderSentiment deep-translator matplotlib
from google_play_scraper import reviews, Sort
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# SCRAPING DATA
result, _ = reviews(
    'com.bpjstku',
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=200
)

df = pd.DataFrame(result)
df = df[['userName', 'score', 'at', 'content']]

df.to_csv('ulasan_jmo.csv', index=False)
print("Data berhasil diambil")

# LOAD MODEL
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

label_map = {0: 'Positif', 1: 'Netral', 2: 'Negatif'}

# ANALISIS SENTIMEN
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

# VISUALISASI
sentiment_counts = df['sentiment'].value_counts()

labels = sentiment_counts.index
values = sentiment_counts.values

colors = ['green', 'gray', 'red']

bars = plt.bar(labels, values, color=colors)

plt.title("Analisis Sentimen JMO Mobile (IndoBERT)")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
             ha='center', va='bottom')

plt.show()
