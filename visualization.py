import matplotlib.pyplot as plt
import os

def plot_sentiment(df):
    counts = df['sentiment'].value_counts()

    labels = counts.index
    values = counts.values

    colors = ['green', 'gray', 'red']

    bars = plt.bar(labels, values, color=colors)

    plt.title("Analisis Sentimen JMO Mobile")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
                 ha='center', va='bottom')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/grafik_sentimen.png')

    plt.show()
