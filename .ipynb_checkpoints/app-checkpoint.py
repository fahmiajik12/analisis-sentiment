import streamlit as st
import pandas as pd
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Model dan tokenizer untuk analisis sentimen
model_id = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Inisialisasi pipeline untuk analisis sentimen menggunakan PyTorch
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework="pt")

# Judul aplikasi
st.title("Dashboard Analisis Sentimen Komentar TikTok")

# Menampilkan video lokal dengan ukuran lebih kecil
video_file = "video.mp4"  # Ganti dengan path ke video lokal Anda
st.video(video_file)
# Menambahkan CSS untuk memperkecil ukuran video
st.markdown("""
    <style>
        .small-video-container {
            width: 50%;  /* Mengurangi lebar video menjadi 50% dari ukuran asli */
            margin: 0 auto;  /* Memastikan video berada di tengah */
        }
    </style>
""", unsafe_allow_html=True)

# Menambahkan link sumber video dengan ukuran font yang lebih kecil
st.markdown("""
    <div class="small-text">
        <a href="https://www.tiktok.com/@pandawaragroup/video/7465971640732650757" target="_blank">
            Sumber Video
        </a>
    </div>
""", unsafe_allow_html=True)

# Fungsi untuk membersihkan teks komentar
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#[\w\d]+", "", text)  # Menghapus URL, mention, dan hashtag
    text = re.sub(r"[^\w\s]", "", text)  # Menghapus karakter selain huruf dan angka
    text = re.sub(r"\d+", "", text)  # Menghapus angka
    return text.strip()

# Upload file JSON dari komentar TikTok
uploaded_file = st.file_uploader("Upload file JSON hasil scraping komentar TikTok", type=["json"])

# Proses data jika file diupload
if uploaded_file:
    raw_data = json.load(uploaded_file)
    df = pd.json_normalize(raw_data)  # Mengubah JSON menjadi DataFrame
    df['clean_text'] = df['text'].astype(str).apply(clean_text)  # Membersihkan teks komentar

    # Prediksi sentimen dengan pipeline
    with st.spinner("Menganalisis sentimen komentar..."):
        results = df['clean_text'].apply(lambda x: sentiment_pipeline(x[:512])[0])
        df['sentiment_label'] = results.apply(lambda x: x['label'])
        df['sentiment_score'] = results.apply(lambda x: x['score'])

    st.success("Analisis selesai!")

    # Menampilkan data komentar dengan hasil sentimen
    st.subheader("Data Komentar dengan Hasil Sentimen")
    st.dataframe(df[['text', 'clean_text', 'sentiment_label', 'sentiment_score']])

    # Filter komentar berdasarkan sentimen
    st.subheader("Filter Komentar")
    selected_label = st.selectbox("Pilih label sentimen:", df['sentiment_label'].unique())
    st.dataframe(df[df['sentiment_label'] == selected_label][['text', 'sentiment_score']])

    # Visualisasi distribusi sentimen
    st.subheader("Distribusi Sentimen")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment_label', palette='Set2', ax=ax)
    st.pyplot(fig)

    # WordCloud per sentimen
    st.subheader("WordCloud per Sentimen")
    for label in df['sentiment_label'].unique():
        text = ' '.join(df[df['sentiment_label'] == label]['clean_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.markdown(f"**{label}**")
        st.image(wordcloud.to_array())

    # Menampilkan komentar dengan skor sentimen tertinggi dan terendah
    st.subheader("Komentar Skor Tertinggi dan Terendah")
    st.markdown("**Skor Tertinggi:**")
    st.dataframe(df.sort_values(by='sentiment_score', ascending=False).head(5)[['text', 'sentiment_label', 'sentiment_score']])

    st.markdown("**Skor Terendah:**")
    st.dataframe(df.sort_values(by='sentiment_score', ascending=True).head(5)[['text', 'sentiment_label', 'sentiment_score']])