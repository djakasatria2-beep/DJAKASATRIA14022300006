from google_play_scraper import reviews, Sort
import pandas as pd
import os

def get_reviews():
    result, _ = reviews(
        'com.bpjstku',
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=200
    )

    df = pd.DataFrame(result)
    df = df[['userName', 'score', 'at', 'content']]

    # simpan ke folder data/raw
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/ulasan_google_play.csv', index=False)

    print("Data berhasil diambil")

    return df
