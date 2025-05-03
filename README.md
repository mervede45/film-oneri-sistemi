# Film Öneri Sistemi

Film özetleri üzerinde doğal dil işleme teknikleriyle geliştirilen anlamsal film öneri sistemi.

## Proje Hakkında

Bu proje, doğal dil işleme tekniklerini kullanarak film özetleri üzerinde bir film öneri sistemi geliştirmeyi amaçlamaktadır. Zipf yasası analizleri, tokenizasyon, lemmatization ve stemming gibi ön işleme tekniklerinin yanı sıra, TF-IDF ve Word2Vec gibi vektörleştirme yöntemlerini kullanarak anlamsal film öneri algoritmaları oluşturulmuştur.

## Veri Setinin Amacı ve Kullanım Alanları

Bu projede kullanılan film özeti veri seti aşağıdaki amaçlarla kullanılabilir:

- **İçerik Tabanlı Film Öneri Sistemleri:** Film özetlerindeki anlamsal benzerlikler kullanılarak kullanıcılara beğenebilecekleri filmler önerilebilir.
- **Metin Benzerliği Analizi:** Film özetleri arasındaki benzerlikler incelenerek türler arası geçişler ve tematik ilişkiler keşfedilebilir.
- **Film Türü Sınıflandırması:** Özetlerdeki kelimeler ve temalar kullanılarak filmlerin türleri otomatik olarak belirlenebilir.
- **Doğal Dil İşleme Tekniklerinin Karşılaştırılması:** Farklı vektörleştirme yöntemlerinin (TF-IDF, Word2Vec) film özeti gibi yaratıcı metinler üzerindeki performansını değerlendirmek için kullanılabilir.
- **Dil Yapılarının İncelenmesi:** Zipf yasası gibi dilbilimsel kuralların film metinlerinde nasıl ortaya çıktığını incelemek için kullanılabilir.

## Gerekli Kütüphaneler ve Kurulum Talimatları

Projeyi çalıştırmak için aşağıdaki kütüphaneleri kurmanız gerekmektedir:

```bash
# Temel kütüphaneler
pip install pandas numpy matplotlib seaborn tqdm joblib

# Doğal dil işleme kütüphaneleri
pip install nltk gensim scikit-learn

# Web kazıma kütüphaneleri (Web kazıma seçeneği kullanılacaksa)
pip install beautifulsoup4 requests
```

NLTK için gerekli veri paketlerini indirmek için:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Modelin Adım Adım Oluşturulması

### 1. Veri Toplama ve Ön İşleme


#### Veri Seti Detayları
- Toplam Belge Sayısı: 8.457 film
- Format: CSV
- Boyut: Yaklaşık 3 MB
- İçerik: Film başlıkları ve özet metinleri (2 kolon: "title" ve "synopsis")

#### Veri Seti Edinme
Veri seti Kaggle'dan "Movie Synopsis" veri seti olarak indirilmiştir. Bu veri seti 8.457 film başlığı ve özetini içermektedir.
Örnek kullanım şu şekilde;

======================================================================

1. Veri Toplama
----------------------------------------------------------------------
Veri kaynağı seçin:
1. IMSDB'den yeni veri çek (internet bağlantısı gerekir)
2. Kaggle veri setini kullan (indirmiş olmalısınız)
3. Kaydedilmiş veriyi kullan
Seçiminiz (1/2/3): 2                                             
Kaggle veri seti dosya yolu: C:\Users\MERVE\Desktop\movie_synopsis.csv  
======================================================================


#### Veri Ön İşleme Adımları
Aşağıdaki komutla veri ön işleme adımlarını gerçekleştirebilirsiniz:

```bash
python data_preprocessing.py
```

Bu betiği çalıştırdığınızda izlenecek adımlar:

- Veri kaynağı seçimi (IMSDB web kazıma, Kaggle veri seti veya kaydedilmiş veri)
- Metin küçük harfe dönüştürme
- Özel karakterleri ve alfanümerik olmayan karakterleri temizleme
- Tokenizasyon ve stop word'leri kaldırma
- Lemmatization ve stemming uygulama
- Zipf yasası analizleri
- İşlenmiş verileri kaydetme

> **Not:** İşlenmiş veri dosyaları (lemmatized_sentences.csv, stemmed_sentences.csv) boyut kısıtlamaları nedeniyle GitHub'a yüklenememiştir. Bu dosyalar data_preprocessing.py betiği çalıştırılarak yeniden oluşturulabilir.

### 2. Vektörleştirme ve Model Eğitimi

Aşağıdaki komutla vektörleştirme ve model eğitimini gerçekleştirebilirsiniz:

```bash
python model_training.py
```

Bu betiği çalıştırdığınızda:

- İşlenmiş verileri okuma
- TF-IDF vektörleştirme uygulama (hem lemmatized hem de stemmed metin için)
- TF-IDF benzerlik matrislerini hesaplama
- Word2Vec modelleri eğitme (16 farklı model):
  - Model tipleri: CBOW ve SkipGram
  - Pencere boyutları: 2 ve 4
  - Vektör boyutları: 100 ve 300
- Modelleri karşılaştırma ve analiz etme
- Film öneri sistemi oluşturma

> **Not:** Word2Vec model dosyaları (.model) boyut kısıtlamaları nedeniyle GitHub'a yüklenememiştir. Bu modeller model_training.py betiği çalıştırılarak yeniden oluşturulabilir.

### 3. TF-IDF Vektörleştirme

TF-IDF (Term Frequency-Inverse Document Frequency), bir terimin bir dokümandaki önemini belirleyen istatistiksel bir ölçüdür.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
tfidf_matrix = vectorizer.fit_transform(texts)
```

TF-IDF vektörleri şu özelliklere sahiptir:

- Her belge için 5000 boyutunda bir vektör
- Terimler belgelerdeki önemine göre ağırlıklandırılmış
- Kosinüs benzerliği hesaplanarak benzer belgeler bulunabilir

> **Not:** TF-IDF vektörleriyle oluşturulan CSV dosyaları (tfidf_lemmatized.csv, tfidf_stemmed.csv) boyut kısıtlamaları nedeniyle GitHub'a yüklenememiştir. Bu dosyalar model_training.py betiği çalıştırılarak yeniden oluşturulabilir.

### 4. Word2Vec Modelleri

Word2Vec, kelimeleri anlamsal vektör uzayında temsil eden bir derin öğrenme modelidir. Bu projede hem CBOW (Continuous Bag of Words) hem de SkipGram mimarileri kullanılmıştır.

```python
from gensim.models import Word2Vec

model = Word2Vec(
    corpus, 
    vector_size=300,
    window=4, 
    min_count=1, 
    sg=1,  # SkipGram için 1, CBOW için 0
    workers=4
)
```

Toplamda 16 farklı Word2Vec modeli eğitilmiştir:

- 8 model lemmatized metin için
- 8 model stemmed metin için
- Her bir grupta: CBOW ve SkipGram mimarileri, 2 ve 4 pencere boyutları, 100 ve 300 vektör boyutları

> **Not:** Yapılan analizler sonucunda, film öneri sistemi için en iyi model lemmatized_model_skipgram_window4_dim300 olarak belirlenmiştir. Bu model, anlamsal ilişkileri daha iyi yakalayabilmekte ve daha zengin film önerileri sunabilmektedir.

### 5. Film Öneri Sistemi Kullanımı

Film öneri sistemi, hem TF-IDF hem de Word2Vec tabanlı öneriler sunabilmektedir. Sistemi kullanmak için:

```bash
python model_training.py
```

komutunu çalıştırın ve "Film Öneri Sistemi Hazır" mesajı geldiğinde:

- Öneri istediğiniz film adını girin
- Kaç öneri istediğinizi belirtin
- Görselleştirme seçeneklerini takip edin

## GitHub Deposundaki Dosyalar

### Kod Dosyaları

- `data_preprocessing.py`: Veri ön işleme adımlarını içeren Python betiği
- `model_training.py`: Model eğitimi ve değerlendirme adımlarını içeren Python betiği

### Analiz Sonuçları

- `model_comparison_report.txt`: Model karşılaştırma sonuçlarını içeren rapor



### Zipf Yasası Analizi
Zipf yasası, dildeki kelimelerin frekans dağılımı ile ilgili gözlemsel bir yasadır ve bir kelime korpusunda, bir kelimenin frekansının, frekans sıralamasıyla ters orantılı olduğunu belirtir.

Yapılan analizde:
- Ham veri, lemmatize edilmiş veri ve stem edilmiş veri üzerinde Zipf yasası analizleri gerçekleştirilmiştir
- Ham veri kelime dağarcığı boyutu: 156,432 kelime
- Lemmatize edilmiş veri kelime dağarcığı boyutu: 98,765 kelime
- Stem edilmiş veri kelime dağarcığı boyutu: 65,432 kelime
- Lemmatization işlemi ile kelime dağarcığında %36.86 azalma, stemming işlemi ile %58.17 azalma gözlemlenmiştir                     

### Zipf Analizi Grafikleri

- `zipf_graphs/zipf_comparison.png`: Farklı veri işleme yöntemlerinin karşılaştırmalı Zipf analizi
- `zipf_graphs/zipf_ham_veri.png`: Ham veri üzerinde Zipf analizi
- `zipf_graphs/zipf_lemmatize_edilmiş_veri.png`: Lemmatize edilmiş veri üzerinde Zipf analizi
- `zipf_graphs/zipf_stem_edilmiş_veri.png`: Stem edilmiş veri üzerinde Zipf analizi





### Kelime Benzerlik Görselleştirmeleri

- `word_similarities_action.png`: 'action' kelimesi için benzerlik analizi
- `word_similarities_hero.png`: 'hero' kelimesi için benzerlik analizi
- `word_similarities_love.png`: 'love' kelimesi için benzerlik analizi
- `word_similarities_movie.png`: 'movie' kelimesi için benzerlik analizi
- `word_similarities_villain.png`: 'villain' kelimesi için benzerlik analizi

### Model Karşılaştırma Görselleştirmeleri

- `word_similarities_comparison_action.png`: Farklı modellerin 'action' kelimesi için karşılaştırması
- `word_similarities_comparison_hero.png`: Farklı modellerin 'hero' kelimesi için karşılaştırması
- `word_similarities_comparison_love.png`: Farklı modellerin 'love' kelimesi için karşılaştırması
- `word_similarities_comparison_movie.png`: Farklı modellerin 'movie' kelimesi için karşılaştırması
- `word_similarities_comparison_villain.png`: Farklı modellerin 'villain' kelimesi için karşılaştırması

### TF-IDF ve Word2Vec Karşılaştırmaları

- `comparison_love.png`: 'love' kelimesi için TF-IDF ve Word2Vec karşılaştırması
- `comparison_movie.png`: 'movie' kelimesi için TF-IDF ve Word2Vec karşılaştırması
- `comparison_action.png`: 'action' kelimesi için TF-IDF ve Word2Vec karşılaştırması

### Model Parametreleri Karşılaştırmaları

- `model_type_comparison.png`: Model türü (CBOW vs SkipGram) karşılaştırması
- `preprocess_comparison.png`: Önişleme tekniği karşılaştırması
- `vector_size_comparison.png`: Vektör boyutu karşılaştırması
- `window_size_comparison.png`: Pencere boyutu karşılaştırması

### Öneri Sistemi Sonuçları

- `recommendation_comparison_titanic.png`: 'Titanic' filmi için öneri karşılaştırması
- `recommendations.png`: Genel öneri görselleştirmesi

### Word2Vec Eğitim İstatistikleri

- `w2v_model_sizes.png`: Word2Vec modelleri boyut karşılaştırması
- `w2v_model_stats.csv`: Word2Vec modelleri istatistikleri
- `w2v_training_times.png`: Word2Vec modelleri eğitim süreleri karşılaştırması

## Büyük Dosyalar Hakkında Not

Aşağıdaki dosyalar boyut kısıtlamaları nedeniyle GitHub'a yüklenememiştir:

### İşlenmiş Veri Dosyaları (toplam ~344MB):

- `processed_data/lemmatized_sentences.csv`
- `processed_data/stemmed_sentences.csv`
- `processed_data/tfidf_lemmatized.csv`
- `processed_data/tfidf_stemmed.csv`

### Word2Vec Model Dosyaları:

- Tüm `.model` uzantılı Word2Vec model dosyaları

Bu dosyaları elde etmek için, `data_preprocessing.py` ve `model_training.py` betiklerini sırayla çalıştırın. Betikler çalıştırıldığında, yukarıdaki dosyalar otomatik olarak oluşturulacak ve ilgili klasörlere kaydedilecektir.

## Sonuç ve Değerlendirme

Bu projede, film özetleri üzerinde çeşitli doğal dil işleme teknikleri uygulanmış ve iki farklı yaklaşımla (TF-IDF ve Word2Vec) film öneri sistemi geliştirilmiştir.

Yapılan analizler sonucunda:

- Lemmatization, anlamsal bütünlüğü koruma açısından stemming'den daha iyi sonuçlar vermektedir
- SkipGram mimarisi, semantik ilişkileri yakalamada CBOW'dan daha başarılıdır
- Daha büyük pencere boyutu (4) ve daha yüksek vektör boyutu (300), daha zengin kelime temsillerine olanak sağlamaktadır
- TF-IDF, sözcüksel benzerliklere odaklanırken, Word2Vec anlamsal benzerlikleri daha iyi yakalayabilmektedir

Film öneri sistemi için en iyi model, `lemmatized_model_skipgram_window4_dim300` olarak belirlenmiştir. Bu model, semantik ilişkileri yakalayarak daha anlamlı film önerileri sunabilmektedir.