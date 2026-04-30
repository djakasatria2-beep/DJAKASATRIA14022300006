# =========================================================
# INSTALL LIBRARY
# =========================================================
!pip install google-play-scraper transformers torch matplotlib pandas

# =========================================================
# IMPORT
# =========================================================
from google_play_scraper import reviews
from transformers import pipeline
import pandas as pd
import csv
import matplotlib.pyplot as plt

# =========================================================
# 1. SCRAPING DATA
# =========================================================
print("Mengambil data ulasan...")

res = reviews(
    'com.bpjstku',
    lang='id',
    country='id',
    count=100
)

result = res[0]

# =========================================================
# 2. LOAD MODEL SENTIMENT
# =========================================================
print("Load model sentiment...")

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier"
)

# =========================================================
# 3. ANALISIS SENTIMEN
# =========================================================
print("Analisis sentimen...")

hasil_final = []

for i, review in enumerate(result):
    print(f"Proses {i+1}/{len(result)}", end='\r')
    
    teks = review['content']
    clean_text = teks[:512] if teks else ""
    
    prediksi = sentiment_pipe(clean_text)[0]
    
    hasil_final.append({
        'userName': review['userName'],
        'score': review['score'],
        'at': review['at'],
        'content': teks,
        'sentiment': prediksi['label'],
        'confidence': round(prediksi['score'], 4)
    })

print("\nSelesai!")

# =========================================================
# 4. SIMPAN KE sample_data
# =========================================================
filename = '/content/sample_data/hasil_sentimen.csv'

with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'userName','score','at','content','sentiment','confidence'
    ])
    writer.writeheader()
    writer.writerows(hasil_final)

print("File disimpan di sample_data")

# =========================================================
# 5. DATAFRAME
# =========================================================
df = pd.DataFrame(hasil_final)
df.head()

# =========================================================
# 6. GRAFIK BERWARNA
# =========================================================
sentiment_counts = df['sentiment'].value_counts()

colors = {
    'positive': 'green',
    'neutral': 'gray',
    'negative': 'red'
}

bar_colors = [colors.get(x, 'blue') for x in sentiment_counts.index]

plt.figure(figsize=(6,4))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)

# label angka di atas batang
for i, v in enumerate(sentiment_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.title('Distribusi Sentimen Ulasan')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah')

plt.show()

# =========================================================
# 7. CEK FILE
# =========================================================
!ls /content/sample_data
