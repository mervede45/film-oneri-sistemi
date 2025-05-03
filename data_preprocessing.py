import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter

# NLTK için gerekli dosyaları indirme
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextProcessor:
    """
    Metinleri işleyen sınıf - Lemmatization, Stemming, TF-IDF ve diğer önişleme adımlarını gerçekleştirir
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def preprocess_text(self, text):
        """
        Metni temizleme ve normalleştirme
        - Küçük harfe çevirme
        - Alfanumerik olmayan karakterleri temizleme
        - Stopword'leri çıkarma
        - Lemmatization ve Stemming uygulanması
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Küçük harfe çevirme
        text = text.lower()
        
        # Alfanumerik olmayan karakterleri kaldırma
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Fazla boşlukları temizleme
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Basit cümle ayırma - nokta, ünlem ve soru işaretlerinden böl
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Her cümle için tokenizasyon, lemmatizasyon ve stemming
        lemmatized_sentences = []
        stemmed_sentences = []
        
        for sentence in sentences:
            # Basit bölme kullanımı
            tokens = sentence.split()
            
            # Stopword'leri çıkarma
            filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
            
            # Lemmatization
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            lemmatized_sentences.append(lemmatized_tokens)
            
            # Stemming
            stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
            stemmed_sentences.append(stemmed_tokens)
        
        # Düz metin olarak birleştirme
        lemmatized_text = ' '.join([' '.join(tokens) for tokens in lemmatized_sentences])
        stemmed_text = ' '.join([' '.join(tokens) for tokens in stemmed_sentences])
        
        return {
            'lemmatized_text': lemmatized_text,
            'stemmed_text': stemmed_text,
            'tokenized_lemmatized': lemmatized_sentences,
            'tokenized_stemmed': stemmed_sentences
        }
    
    def create_tfidf_vectors(self, texts, max_features=5000, min_df=2, max_df=0.85):
        """
        Metinler için TF-IDF vektörlerini oluşturma
        
        Parameters:
        -----------
        texts: list
            İşlenmiş metinlerin listesi (lemmatized veya stemmed)
        max_features: int, default=5000
            TF-IDF vektörleştirmede kullanılacak maksimum özellik sayısı
        min_df: int or float, default=2
            Minimum belge frekansı
        max_df: float, default=0.85
            Maksimum belge frekansı
            
        Returns:
        --------
        dict
            TF-IDF matrisini ve özellik adlarını içeren sözlük
        """
        # TF-IDF vektörleştiriciyi başlatma
        self.vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
        
        # TF-IDF matrisini oluşturma
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Özellik adlarını alma
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return {
            'tfidf_matrix': self.tfidf_matrix,
            'feature_names': self.feature_names,
            'vectorizer': self.vectorizer
        }
    
    def get_top_tfidf_words(self, text_index, top_n=5):
        """
        Belirli bir metin için en yüksek TF-IDF skoruna sahip kelimeleri bulma
        
        Parameters:
        -----------
        text_index: int
            Metnin indeksi
        top_n: int, default=5
            Döndürülecek en yüksek skorlu kelime sayısı
            
        Returns:
        --------
        list
            Yüksek TF-IDF skoruna sahip kelimeler ve skorlarının listesi
        """
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("Önce create_tfidf_vectors metodunu çağırmalısınız.")
        
        # Metnin TF-IDF vektörünü alma
        text_vector = self.tfidf_matrix[text_index].toarray().flatten()
        
        # Skorlara göre sıralama
        top_indices = text_vector.argsort()[-top_n:][::-1]
        
        # Kelimeleri ve skorları alma
        top_words = [(self.feature_names[i], text_vector[i]) for i in top_indices]
        
        return top_words
    
    def find_similar_words(self, word, top_n=5):
        """
        TF-IDF vektörlerine dayalı olarak bir kelimeye benzer kelimeleri bulma
        
        Parameters:
        -----------
        word: str
            Benzerliği bulunacak kelime
        top_n: int, default=5
            Döndürülecek benzer kelime sayısı
            
        Returns:
        --------
        list
            Benzer kelimeler ve benzerlik skorlarının listesi
        """
        if self.tfidf_matrix is None or self.feature_names is None:
            raise ValueError("Önce create_tfidf_vectors metodunu çağırmalısınız.")
        
        try:
            # Kelimenin indeksini bulma
            word_index = self.feature_names.tolist().index(word.lower())
            
            # Kelimenin TF-IDF vektörünü alma
            word_vector = self.tfidf_matrix[:, word_index].toarray().T
            
            # Tüm kelimelerin TF-IDF vektörlerini alma
            tfidf_vectors = self.tfidf_matrix.toarray().T
            
            # Kosinüs benzerliğini hesaplama
            similarities = cosine_similarity(word_vector, tfidf_vectors)
            similarities = similarities.flatten()
            
            # En yüksek benzerlik skorlarını bulma (kendisi hariç)
            indices = similarities.argsort()[-(top_n+1):][::-1]
            
            # Benzer kelimeleri döndürme
            similar_words = [(self.feature_names[i], similarities[i]) for i in indices if i != word_index][:top_n]
            
            return similar_words
        except ValueError:
            print(f"'{word}' kelimesi vektörleştirme sözlüğünde bulunamadı.")
            return []
    
    def calculate_sentence_similarity(self, sentence1_idx, sentence2_idx):
        """
        İki cümle arasındaki TF-IDF tabanlı kosinüs benzerliğini hesaplama
        
        Parameters:
        -----------
        sentence1_idx: int
            İlk cümlenin indeksi
        sentence2_idx: int
            İkinci cümlenin indeksi
            
        Returns:
        --------
        float
            İki cümle arasındaki benzerlik skoru
        """
        if self.tfidf_matrix is None:
            raise ValueError("Önce create_tfidf_vectors metodunu çağırmalısınız.")
        
        # Cümle vektörlerini alma
        vec1 = self.tfidf_matrix[sentence1_idx].toarray().flatten()
        vec2 = self.tfidf_matrix[sentence2_idx].toarray().flatten()
        
        # Kosinüs benzerliğini hesaplama
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        
        return similarity

def zipf_analizi_ciz(metin, baslik, output_dir="zipf_graphs"):
    """
    Verilen metin için Zipf yasası analizini yaparak log-log grafiğini çizer
    
    Parameters:
    -----------
    metin: str
        Analiz edilecek metin
    baslik: str
        Grafiğin başlığı
    output_dir: str, default="zipf_graphs"
        Grafiklerin kaydedileceği dizin
        
    Returns:
    --------
    tuple
        (kelimeler, sıralar, frekanslar) içeren tuple
    """
    # Çıktı dizinini oluşturma
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Kelime frekanslarını say
    kelime_sayimlari = Counter(metin.split())
    
    # Sıraları ve frekansları al
    kelimeler = sorted(kelime_sayimlari.items(), key=lambda x: x[1], reverse=True)
    siralar = np.arange(1, len(kelimeler) + 1)
    frekanslar = [sayi for kelime, sayi in kelimeler]
    
    # Log-log grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.loglog(siralar, frekanslar, marker='.', linestyle='none', alpha=0.5)
    
    # Regresyon çizgisi ekle
    plt.loglog(siralar, [frekanslar[0]/r for r in siralar], linestyle='-', color='r', 
              label='Zipf Yasası (1/rank)')
    
    # Etiket ve başlık ekle
    plt.xlabel('Sıra (log ölçeği)')
    plt.ylabel('Frekans (log ölçeği)')
    plt.title(f'Zipf Dağılımı: {baslik}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Grafiği kaydet
    plt.savefig(os.path.join(output_dir, f'zipf_{baslik.lower().replace(" ", "_")}.png'))
    plt.close()
    
    print(f"{baslik} için Zipf analizi grafiği kaydedildi.")
    
    # İstatistikler
    top_words = kelimeler[:20]
    print(f"\nEn sık kullanılan 20 kelime ({baslik}):")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"{i}. {word}: {count}")
    
    return kelimeler, siralar, frekanslar

# Veri Toplama Fonksiyonları
def scrape_imsdb(limit=100):
    """
    IMSDB'den film senaryolarını çekme
    """
    base_url = "https://imsdb.com/all-scripts.html"
    scripts_data = []
    
    try:
        # Ana sayfayı çekme
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tüm senaryo linklerini bulma
        script_links = []
        for a_tag in soup.find_all('a'):
            href = a_tag.get('href', '')
            if href.startswith('/Movie Scripts/'):
                script_links.append(f"https://imsdb.com{href}")
        
        # Limit kadar senaryo çekme
        for i, link in enumerate(script_links[:limit]):
            try:
                script_response = requests.get(link)
                script_soup = BeautifulSoup(script_response.content, 'html.parser')
                
                # Film adını çekme
                title = script_soup.title.text.replace(' Script at IMSDb.', '')
                
                # Senaryo metnini çekme
                script_text = ""
                pre_tags = script_soup.find_all('pre')
                if pre_tags:
                    for pre in pre_tags:
                        script_text += pre.text
                
                if script_text:
                    scripts_data.append({
                        "title": title,
                        "script": script_text
                    })
                    print(f"Scraped: {title}")
            except Exception as e:
                print(f"Error scraping {link}: {e}")
                
    except Exception as e:
        print(f"Error accessing IMSDB: {e}")
    
    return pd.DataFrame(scripts_data)

def get_kaggle_movie_synopsis(dataset_path="movie_synopsis.csv"):
    """
    Kaggle'dan film özet verilerini okuma (indirildikten sonra)
    """
    try:
        df = pd.read_csv(dataset_path)
        print(f"Kaggle veri seti yüklendi: {len(df)} kayıt")
        return df
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
        return pd.DataFrame()

def extract_dialogues(script_text):
    """
    Senaryodan diyalogları çıkarma
    """
    # Basit bir yaklaşım: Tırnak içindeki metinleri diyalog olarak kabul etme
    dialogue_pattern = r'"([^"]*)"'
    dialogues = re.findall(dialogue_pattern, script_text)
    
    # Alternatif yaklaşım: Karakter adı ve diyalog formatını tanıma
    char_dialogue_pattern = r'([A-Z][A-Z\s]+)(?:\s*\(.*\))?\s*\n([\s\S]*?)(?=\n\s*\n|\n[A-Z][A-Z\s]+|\Z)'
    char_dialogues = re.findall(char_dialogue_pattern, script_text)
    
    # Her iki yaklaşımın sonuçlarını birleştirme
    all_dialogues = dialogues + [dialogue for _, dialogue in char_dialogues]
    
    return ' '.join(all_dialogues)

def process_movie_data(movie_data, output_dir="processed_data"):
    """
    Film verilerini işleme ve kaydetme
    """
    # Çıktı dizinini oluşturma
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # TextProcessor nesnesi oluşturma
    text_processor = TextProcessor()
    
    # Uygun metin sütununu belirleme
    columns = movie_data.columns.tolist()
    print(f"Veri setindeki sütunlar: {columns}")
    
    # Metin için uygun sütunu seçme
    if 'script' in columns:
        text_column = 'script'
        movie_data['text_for_processing'] = movie_data['script'].apply(extract_dialogues)
    elif 'synopsis' in columns:
        text_column = 'synopsis'
        movie_data['text_for_processing'] = movie_data['synopsis']
    else:
        # Kapsayıcı bir yaklaşım: İçerik için olası sütunları arama
        text_columns = [col for col in columns if any(
            keyword in col.lower() for keyword in ['synopsis', 'plot', 'summary', 'description', 'overview', 'text', 'content']
        )]
        
        if text_columns:
            text_column = text_columns[0]
            print(f"Metin içerik sütunu olarak '{text_column}' kullanılıyor.")
            movie_data['text_for_processing'] = movie_data[text_column]
        else:
            raise ValueError("Uyumlu metin sütunu bulunamadı.")
    
    # İşlenmiş verileri saklamak için yeni DataFrame oluşturma
    processed_data = pd.DataFrame({
        'title': movie_data['title'],
        'original_text': movie_data['text_for_processing']
    })
    
    # Ham metin üzerinde Zipf analizi
    print("\nHam metin üzerinde Zipf yasası analizi yapılıyor...")
    all_raw_text = ' '.join(movie_data['text_for_processing'].fillna('').astype(str))
    raw_words, raw_ranks, raw_freqs = zipf_analizi_ciz(all_raw_text, "Ham Veri")
    
    # Lemmatized ve stemmed metinler için listeler
    lemmatized_texts = []
    stemmed_texts = []
    
    # Her metin için işleme
    print("Metinler işleniyor (Lemmatizasyon ve Stemming uygulanıyor)...")
    for idx, row in tqdm(movie_data.iterrows(), total=len(movie_data), desc="İşleniyor"):
        text = str(row['text_for_processing'])
        processed = text_processor.preprocess_text(text)
        
        lemmatized_texts.append(processed['lemmatized_text'])
        stemmed_texts.append(processed['stemmed_text'])
    
    # İşlenmiş metinleri DataFrame'e ekleme
    processed_data['lemmatized_text'] = lemmatized_texts
    processed_data['stemmed_text'] = stemmed_texts
    
    # İşlenmiş metin üzerinde Zipf analizi
    print("\nLemmatize edilmiş metin üzerinde Zipf yasası analizi yapılıyor...")
    all_lemmatized_text = ' '.join(processed_data['lemmatized_text'].fillna('').astype(str))
    lemma_words, lemma_ranks, lemma_freqs = zipf_analizi_ciz(all_lemmatized_text, "Lemmatize Edilmiş Veri")
    
    print("\nStem edilmiş metin üzerinde Zipf yasası analizi yapılıyor...")
    all_stemmed_text = ' '.join(processed_data['stemmed_text'].fillna('').astype(str))
    stem_words, stem_ranks, stem_freqs = zipf_analizi_ciz(all_stemmed_text, "Stem Edilmiş Veri")
    
    # Zipf analizleri karşılaştırma grafiği
    plt.figure(figsize=(12, 8))
    plt.loglog(raw_ranks[:1000], raw_freqs[:1000], 'b.', alpha=0.5, label='Ham Veri')
    plt.loglog(lemma_ranks[:1000], lemma_freqs[:1000], 'g.', alpha=0.5, label='Lemmatize Edilmiş')
    plt.loglog(stem_ranks[:1000], stem_freqs[:1000], 'r.', alpha=0.5, label='Stem Edilmiş')
    
    # Teorik Zipf eğrisi
    plt.loglog(raw_ranks[:1000], [raw_freqs[0]/r for r in raw_ranks[:1000]], 'k-', 
              label='Zipf Yasası (1/rank)')
    
    plt.xlabel('Sıra (log ölçeği)')
    plt.ylabel('Frekans (log ölçeği)')
    plt.title('Zipf Yasası Karşılaştırması: Ham vs. İşlenmiş Veri')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join("zipf_graphs", "zipf_comparison.png"))
    plt.close()
    
    # Kelime dağarcığı boyutları karşılaştırma
    print("\nKelime dağarcığı karşılaştırması:")
    print(f"Ham kelime dağarcığı boyutu: {len(raw_words)}")
    print(f"Lemmatize edilmiş kelime dağarcığı boyutu: {len(lemma_words)}")
    print(f"Stem edilmiş kelime dağarcığı boyutu: {len(stem_words)}")
    print(f"Kelime dağarcığı azalması (lemma): {(1 - len(lemma_words)/len(raw_words))*100:.2f}%")
    print(f"Kelime dağarcığı azalması (stem): {(1 - len(stem_words)/len(raw_words))*100:.2f}%")
    
    # TF-IDF vektörlerini oluşturma
    print("\nTF-IDF vektörleri oluşturuluyor...")
    tfidf_result = text_processor.create_tfidf_vectors(lemmatized_texts)
    
    # İlk birkaç metin için en yüksek TF-IDF skoruna sahip kelimeleri görüntüleme
    num_samples = min(5, len(processed_data))
    print(f"\nİlk {num_samples} metin için en yüksek TF-IDF skoruna sahip kelimeler:")
    for i in range(num_samples):
        top_words = text_processor.get_top_tfidf_words(i, top_n=5)
        print(f"Film: {processed_data.iloc[i]['title']}")
        for word, score in top_words:
            print(f"  {word}: {score:.4f}")
        print()
    
    # CSV dosyalarını oluşturma
    # Ana işlenmiş veri
    processed_data.to_csv(os.path.join(output_dir, "processed_movies.csv"), index=False)
    
    # Lemmatized metinler
    with open(os.path.join(output_dir, "lemmatized_sentences.csv"), mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['title', 'lemmatized_text'])  # Başlık satırı
        for idx, row in processed_data.iterrows():
            writer.writerow([row['title'], row['lemmatized_text']])
    
    # Stemmed metinler
    with open(os.path.join(output_dir, "stemmed_sentences.csv"), mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(['title', 'stemmed_text'])  # Başlık satırı
        for idx, row in processed_data.iterrows():
            writer.writerow([row['title'], row['stemmed_text']])
    
    # TF-IDF DataFrame oluşturma ve kaydetme
    # Lemmatized metinler için TF-IDF DataFrame
    lemmatized_tfidf_df = pd.DataFrame(
        tfidf_result['tfidf_matrix'].toarray(),
        columns=tfidf_result['feature_names'],
        index=processed_data['title']
    )
    lemmatized_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_lemmatized.csv"))
    
    # Stemmed metinler için TF-IDF DataFrame
    stemmed_tfidf_result = text_processor.create_tfidf_vectors(stemmed_texts)
    stemmed_tfidf_df = pd.DataFrame(
        stemmed_tfidf_result['tfidf_matrix'].toarray(),
        columns=stemmed_tfidf_result['feature_names'],
        index=processed_data['title']
    )
    stemmed_tfidf_df.to_csv(os.path.join(output_dir, "tfidf_stemmed.csv"))
    
    # TF-IDF vektörleştiriciyi kaydetme
    models_dir = "models"  # Ana models klasörü
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    joblib_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    import joblib
    joblib.dump(tfidf_result['vectorizer'], joblib_path)
    print(f"TF-IDF vektörleştirici '{joblib_path}' dosyasına kaydedildi.")
    
    print(f"İşlenmiş veriler '{output_dir}' dizinine kaydedildi.")
    print(f"Toplam {len(processed_data)} film işlendi.")
    
    return processed_data, tfidf_result

def main():
    """
    Ana işlevi çalıştırma
    """
    print("Film ve Dizi Senaryosu Veri Ön İşleme")
    print("=" * 70)
    
    # 1. Veri Toplama
    print("\n1. Veri Toplama")
    print("-" * 70)
    
    # Kullanıcıdan veri kaynağı seçmesini isteme
    print("Veri kaynağı seçin:")
    print("1. IMSDB'den yeni veri çek (internet bağlantısı gerekir)")
    print("2. Kaggle veri setini kullan (indirmiş olmalısınız)")
    print("3. Kaydedilmiş veriyi kullan")
    
    choice = input("Seçiminiz (1/2/3): ")
    
    if choice == '1':
        limit = int(input("Kaç film senaryosu çekilsin? (önerilen: 50-100): "))
        df = scrape_imsdb(limit=limit)
        if not df.empty:
            df.to_csv('raw_movie_scripts.csv', index=False)
            print(f"{len(df)} film senaryosu kaydedildi.")
    
    elif choice == '2':
        path = input("Kaggle veri seti dosya yolu: ")
        df = get_kaggle_movie_synopsis(dataset_path=path)
        if not df.empty:
            print(f"{len(df)} film verisi yüklendi.")
    
    elif choice == '3':
        path = input("Kaydedilmiş veri dosya yolu: ")
        try:
            df = pd.read_csv(path)
            print(f"{len(df)} film verisi yüklendi.")
        except Exception as e:
            print(f"Hata: {e}")
            return
    
    else:
        print("Geçersiz seçim!")
        return
    
    # 2. Veri İşleme
    print("\n2. Veri İşleme")
    print("-" * 70)
    
    # Çıktı dizini belirleme
    output_dir = input("İşlenmiş verileri kaydetmek için dizin adı (varsayılan: processed_data): ") or "processed_data"
    
    # Veriyi işleme ve kaydetme
    processed_data, tfidf_result = process_movie_data(df, output_dir=output_dir)
    
    # 3. TF-IDF Analizi
    print("\n3. TF-IDF Analizi")
    print("-" * 70)
    
    # TextProcessor nesnesi
    text_processor = TextProcessor()
    text_processor.vectorizer = tfidf_result['vectorizer']
    text_processor.tfidf_matrix = tfidf_result['tfidf_matrix']
    text_processor.feature_names = tfidf_result['feature_names']
    
    # Benzer kelime analizi
    while True:
        word = input("\nBenzer kelimeler aramak için bir kelime girin (çıkmak için 'q'): ")
        
        if word.lower() == 'q':
            break
        
        similar_words = text_processor.find_similar_words(word, top_n=5)
        
        if similar_words:
            print(f"'{word}' kelimesine en benzer 5 kelime:")
            for w, score in similar_words:
                print(f"  {w}: {score:.4f}")
    
    print("\nVeri ön işleme tamamlandı.")
    print(f"İşlenmiş veriler '{output_dir}' dizininde saklanıyor.")
    print("Bu verileri kullanarak 'model_training.py' ile model eğitimi yapabilirsiniz.")

if __name__ == "__main__":
    main()