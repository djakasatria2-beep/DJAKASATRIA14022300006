from src.scraper import get_reviews
from src.sentiment import analyze_sentiment
from src.visualization import plot_sentiment

# Ambil data
df = get_reviews()

# Analisis sentimen
df = analyze_sentiment(df)

# Visualisasi
plot_sentiment(df)

print("Proses selesai!")
