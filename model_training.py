import os
import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.cm as cm

class TfidfAnalyzer:
    """TF-IDF analizi için sınıf"""
    def __init__(self, texts=None, max_features=5000, min_df=2, max_df=0.85):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.texts = texts
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.similarity_matrix = None
        
        if texts is not None:
            self.create_tfidf_vectors(texts)
    
    def create_tfidf_vectors(self, texts):
        """
        Metinler için TF-IDF vektörlerini oluşturma
        """
        print("TF-IDF vektörleri oluşturuluyor...")
        
        self.texts = texts
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, 
                                         min_df=self.min_df, 
                                         max_df=self.max_df)
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f"TF-IDF matrisi oluşturuldu. Şekil: {self.tfidf_matrix.shape}")
            
            # Benzerlik matrisini de oluşturalım
            self.calculate_similarity_matrix()
            
            return self.tfidf_matrix
        except ValueError as e:
            print(f"TF-IDF vektörleri oluşturulurken hata: {e}")
            return None
    
    def calculate_similarity_matrix(self):
        """
        TF-IDF vektörleri arasında benzerlik matrisini hesaplama
        """
        if self.tfidf_matrix is None:
            raise ValueError("Önce create_tfidf_vectors metodunu çağırmalısınız.")
            
        print("Benzerlik matrisi hesaplanıyor...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"Benzerlik matrisi oluşturuldu. Şekil: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def get_top_tfidf_words(self, text_index, top_n=5):
        """
        Belirli bir metin için en yüksek TF-IDF skoruna sahip kelimeleri bulma
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
    
    def find_similar_documents(self, doc_index, top_n=5):
        """
        Bir belgeye benzer belgeleri bulma
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        # Benzerlik skorlarını alma
        sim_scores = list(enumerate(self.similarity_matrix[doc_index]))
        
        # Benzerlik skorlarına göre sıralama
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Kendisini çıkartma
        sim_scores = sim_scores[1:top_n+1]
        
        # Belge indekslerini ve skorlarını döndürme
        similar_docs = [(i, score) for i, score in sim_scores]
        
        return similar_docs
    
    def find_similar_words(self, word, top_n=5):
        """
        TF-IDF vektörlerine dayalı olarak bir kelimeye benzer kelimeleri bulma
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
    
    def visualize_word_similarities(self, word, similar_words):
        """
        Bir kelimeye benzer kelimeleri görselleştirme
        """
        if not similar_words:
            print(f"'{word}' için benzer kelime bulunamadı.")
            return
        
        words = [w for w, _ in similar_words]
        scores = [s for _, s in similar_words]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=scores, y=words)
        plt.title(f"'{word}' Kelimesine Benzer Kelimeler")
        plt.xlabel('Benzerlik Skoru')
        plt.ylabel('Kelimeler')
        plt.tight_layout()
        plt.savefig(f"word_similarities_{word}.png")
        plt.close()
    
    def save_model(self, output_dir="models"):
        """TF-IDF modelini kaydetme"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Vektörleştiriciyi kaydetme
        vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
        joblib.dump(self.vectorizer, vectorizer_path)
    
        # Benzerlik matrisini kaydetme
        if self.similarity_matrix is not None:
            similarity_path = os.path.join(output_dir, "tfidf_similarity_matrix.pkl")
            joblib.dump(self.similarity_matrix, similarity_path)
    
        print(f"TF-IDF modeli '{output_dir}' dizinine kaydedildi.")
    
    @classmethod
    def load_model(cls, vectorizer_path, texts=None):
        """Kaydedilmiş TF-IDF modelini yükleme"""
        
        # Vektörleştiriciyi yükleme
        vectorizer = joblib.load(vectorizer_path)
        
        # Yeni TfidfAnalyzer örneği oluşturma
        analyzer = cls()
        analyzer.vectorizer = vectorizer
        analyzer.feature_names = vectorizer.get_feature_names_out()
        
        # Eğer metinler verildiyse, TF-IDF matrisini oluşturma
        if texts is not None:
            analyzer.texts = texts
            analyzer.tfidf_matrix = vectorizer.transform(texts)
            analyzer.calculate_similarity_matrix()
        
        print("TF-IDF modeli başarıyla yüklendi.")
        return analyzer


class Word2VecTrainer:
    """Word2Vec model eğitimi için sınıf"""
    def __init__(self):
        self.parameters = [
            {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
            {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
            {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
            {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
            {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
            {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
            {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
            {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
        ]
        self.models = {}
        self.training_stats = {}
    
    def train_and_save_model(self, corpus, params, model_name, output_dir="models"):
        """Word2Vec modelini eğitme ve kaydetme"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Eğitim süresini ölçme
        start_time = time.time()
        
        model = Word2Vec(
            corpus, 
            vector_size=params['vector_size'],
            window=params['window'], 
            min_count=1, 
            sg=1 if params['model_type'] == 'skipgram' else 0,
            workers=4  # Çok çekirdekli işlemcilerde daha hızlı eğitim
        )
        
        training_time = time.time() - start_time
        
        model_path = os.path.join(output_dir, f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model")
        model.save(model_path)
        
        # Model boyutunu hesaplama (MB cinsinden)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        print(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']} model saved!")
        
        # İstatistikleri kaydetme
        stats = {
            'model_type': params['model_type'],
            'window': params['window'],
            'vector_size': params['vector_size'],
            'training_time': training_time,
            'vocabulary_size': len(model.wv.index_to_key),
            'model_size_mb': model_size_mb
        }
        
        full_model_name = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}"
        self.training_stats[full_model_name] = stats
        self.models[full_model_name] = model
        
        return model
    
    def train_all_models(self, lemmatized_corpus, stemmed_corpus, output_dir="models"):
        """Lemmatize ve stem edilmiş corpus'lar için tüm modelleri eğitme"""
        models = {}
        
        # Lemmatize edilmiş corpus ile modelleri eğitme
        print("Lemmatize edilmiş metin üzerinde modeller eğitiliyor...")
        for param in self.parameters:
            model = self.train_and_save_model(lemmatized_corpus, param, "lemmatized_model", output_dir)
            models[f"lemmatized_{param['model_type']}_w{param['window']}_d{param['vector_size']}"] = model
        
        # Stemlenmiş corpus ile modelleri eğitme
        print("Stem edilmiş metin üzerinde modeller eğitiliyor...")
        for param in self.parameters:
            model = self.train_and_save_model(stemmed_corpus, param, "stemmed_model", output_dir)
            models[f"stemmed_{param['model_type']}_w{param['window']}_d{param['vector_size']}"] = model
        
        return models
    
    def get_similar_words(self, model, word, top_n=5):
        """Bir kelimeye en benzer kelimeleri bulmak"""
        try:
            similar_words = model.wv.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            print(f"'{word}' kelimesi model vokabülerinde bulunamadı.")
            return []
    
    def visualize_training_stats(self):
        """Eğitim istatistiklerini görselleştirme"""
        if not self.training_stats:
            print("Henüz model eğitim istatistikleri mevcut değil.")
            return
        
        # İstatistikleri DataFrame'e dönüştürme
        stats_df = pd.DataFrame.from_dict(self.training_stats, orient='index')
        
        # Eğitim sürelerini görselleştirme
        plt.figure(figsize=(14, 7))
        sns.barplot(x=stats_df.index, y='training_time', data=stats_df)
        plt.title('Word2Vec Modelleri Eğitim Süreleri')
        plt.xlabel('Model')
        plt.ylabel('Eğitim Süresi (saniye)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("w2v_training_times.png")
        plt.close()
        
        # Model boyutlarını görselleştirme
        plt.figure(figsize=(14, 7))
        sns.barplot(x=stats_df.index, y='model_size_mb', data=stats_df)
        plt.title('Word2Vec Modelleri Boyutları')
        plt.xlabel('Model')
        plt.ylabel('Model Boyutu (MB)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("w2v_model_sizes.png")
        plt.close()
        
        # İstatistikleri csv olarak kaydetme
        stats_df.to_csv("w2v_model_stats.csv")
        
        print("Eğitim istatistikleri görselleştirildi ve w2v_model_stats.csv dosyasına kaydedildi.")
        
        return stats_df


class ModelComparator:
    """Word2Vec modellerini karşılaştırmak için sınıf"""
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.model_stats = {}
        self.test_words = ["movie", "love", "action", "hero", "villain", "find", "life"]
        self.similarity_results = {}
        
    def load_models(self, model_patterns=None):
        """
        Belirtilen dizindeki modelleri yükleme
        
        Parameters:
        -----------
        model_patterns: list, optional
            Yüklenecek model dosya adı kalıpları. None ise tüm .model dosyaları yüklenir.
        """
        print("Modeller yükleniyor...")
        
        # Dizindeki tüm .model dosyalarını bulma
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.model')]
        
        # Eğer belirli kalıplar belirtilmişse, sadece onları yükle
        if model_patterns:
            filtered_files = []
            for pattern in model_patterns:
                filtered_files.extend([f for f in model_files if pattern in f])
            model_files = filtered_files
        
        # Modelleri yükleme
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            try:
                model = Word2Vec.load(model_path)
                self.models[model_file] = model
                
                # Model istatistiklerini kaydet
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                
                # Model adından parametreleri çıkarma
                if "lemmatized" in model_file:
                    preprocess = "lemmatized"
                else:
                    preprocess = "stemmed"
                
                if "cbow" in model_file:
                    model_type = "cbow"
                else:
                    model_type = "skipgram"
                
                window = int(re.search(r'window(\d+)', model_file).group(1))
                dim = int(re.search(r'dim(\d+)', model_file).group(1))
                
                self.model_stats[model_file] = {
                    'preprocess': preprocess,
                    'model_type': model_type,
                    'window': window,
                    'vector_size': dim,
                    'vocabulary_size': len(model.wv.index_to_key),
                    'model_size_mb': model_size_mb
                }
                
                print(f"Yüklendi: {model_file}")
            except Exception as e:
                print(f"Hata: {model_file} yüklenemedi - {e}")
        
        print(f"Toplam {len(self.models)} model yüklendi.")

    def visualize_word_similarities(self, word, top_n=5):
        """
        Bir kelimeye benzer kelimeleri farklı modeller için görselleştirme
        
        Parameters:
        -----------
        word: str
            Benzerliği görselleştirilecek kelime
        top_n: int, default=5
            Her model için görselleştirilecek benzer kelime sayısı
        """
        plt.figure(figsize=(15, 10))
        
        # Kaç model var?
        n_models = len(self.models)
        cols = 2
        rows = (n_models + 1) // cols
        
        for i, (model_name, model) in enumerate(self.models.items(), 1):
            try:
                # Benzer kelimeleri bulma
                similar_words = model.wv.most_similar(word, topn=top_n)
                words = [w for w, _ in similar_words]
                scores = [s for _, s in similar_words]
                
                # Alt grafik oluşturma
                plt.subplot(rows, cols, i)
                sns.barplot(x=scores, y=words)
                plt.title(f"{model_name}")
                plt.xlabel('Benzerlik Skoru')
                plt.xlim(0, 1)  # Benzerlik skorları 0-1 arasında
            except KeyError:
                plt.subplot(rows, cols, i)
                plt.text(0.5, 0.5, f"'{word}' kelimesi bulunamadı", 
                        ha='center', va='center', fontsize=12)
                plt.title(f"{model_name}")
                plt.axis('off')
        
        plt.suptitle(f"'{word}' Kelimesine Benzer Kelimeler - Model Karşılaştırması", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"word_similarities_comparison_{word}.png")
        plt.close()
        
        print(f"'{word}' kelimesi için model karşılaştırması görselleştirildi.")
    
    def compare_similar_words(self, test_words=None, top_n=5):
        """
        Farklı modellerde belirli kelimelerin benzerliklerini karşılaştırma
        
        Parameters:
        -----------
        test_words: list, optional
            Test edilecek kelimeler. None ise varsayılan kelimeler kullanılır.
        top_n: int, default=5
            Her kelime için döndürülecek benzer kelime sayısı
        """
        if not test_words:
            test_words = self.test_words
        
        print(f"Test kelimeleri: {test_words}")
        print("Kelime benzerlikleri karşılaştırılıyor...")
        
        similarity_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nModel: {model_name}")
            model_results = {}
            
            for word in test_words:
                try:
                    similar_words = model.wv.most_similar(word, topn=top_n)
                    model_results[word] = similar_words
                    
                    print(f"  '{word}' için en benzer {top_n} kelime:")
                    for w, score in similar_words:
                        print(f"    {w}: {score:.4f}")
                except KeyError:
                    print(f"  '{word}' kelimesi model vokabülerinde bulunamadı.")
                    model_results[word] = []
            
            similarity_results[model_name] = model_results
        
        self.similarity_results = similarity_results
        return similarity_results
    
    def compare_model_parameters(self):
        """
        Model parametrelerinin performans üzerindeki etkisini analiz eder
        
        Returns:
        --------
        pandas.DataFrame
            Model parametreleri ve istatistikleri içeren DataFrame
        """
        if not self.model_stats:
            print("Karşılaştırma için yeterli veri bulunmuyor. Önce modelleri yükleyin.")
            return None
        
        print("\nModel parametre karşılaştırması yapılıyor...")
        
        # Model istatistiklerini DataFrame'e dönüştür
        stats_df = pd.DataFrame.from_dict(self.model_stats, orient='index')
        
        # 1. Model türü etkisi (CBOW vs SkipGram)
        print("\n1. Model Türü Etkisi (CBOW vs SkipGram):")
        model_type_stats = stats_df.groupby('model_type')[['vocabulary_size', 'model_size_mb']].mean()
        print(model_type_stats)
        
        # Görselleştirme
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=model_type_stats.index, y='vocabulary_size', data=model_type_stats)
        plt.title('Model Türü - Ortalama Vokabüler Boyutu')
        plt.ylabel('Vokabüler Boyutu')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=model_type_stats.index, y='model_size_mb', data=model_type_stats)
        plt.title('Model Türü - Ortalama Model Boyutu (MB)')
        plt.ylabel('Model Boyutu (MB)')
        
        plt.tight_layout()
        plt.savefig("model_type_comparison.png")
        plt.close()
        
        # 2. Pencere boyutu etkisi
        print("\n2. Pencere Boyutu Etkisi:")
        window_stats = stats_df.groupby('window')[['vocabulary_size', 'model_size_mb']].mean()
        print(window_stats)
        
        # Görselleştirme
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=window_stats.index, y='vocabulary_size', data=window_stats)
        plt.title('Pencere Boyutu - Ortalama Vokabüler Boyutu')
        plt.ylabel('Vokabüler Boyutu')
        plt.xlabel('Pencere Boyutu')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=window_stats.index, y='model_size_mb', data=window_stats)
        plt.title('Pencere Boyutu - Ortalama Model Boyutu (MB)')
        plt.ylabel('Model Boyutu (MB)')
        plt.xlabel('Pencere Boyutu')
        
        plt.tight_layout()
        plt.savefig("window_size_comparison.png")
        plt.close()
        
        # 3. Vektör boyutu etkisi
        print("\n3. Vektör Boyutu Etkisi:")
        vector_size_stats = stats_df.groupby('vector_size')[['vocabulary_size', 'model_size_mb']].mean()
        print(vector_size_stats)
        
        # Görselleştirme
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=vector_size_stats.index, y='vocabulary_size', data=vector_size_stats)
        plt.title('Vektör Boyutu - Ortalama Vokabüler Boyutu')
        plt.ylabel('Vokabüler Boyutu')
        plt.xlabel('Vektör Boyutu')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=vector_size_stats.index, y='model_size_mb', data=vector_size_stats)
        plt.title('Vektör Boyutu - Ortalama Model Boyutu (MB)')
        plt.ylabel('Model Boyutu (MB)')
        plt.xlabel('Vektör Boyutu')
        
        plt.tight_layout()
        plt.savefig("vector_size_comparison.png")
        plt.close()
        
        # 4. Önişleme tekniği etkisi
        print("\n4. Önişleme Tekniği Etkisi (Lemmatize vs Stem):")
        preprocess_stats = stats_df.groupby('preprocess')[['vocabulary_size', 'model_size_mb']].mean()
        print(preprocess_stats)
        
        # Görselleştirme
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=preprocess_stats.index, y='vocabulary_size', data=preprocess_stats)
        plt.title('Önişleme Tekniği - Ortalama Vokabüler Boyutu')
        plt.ylabel('Vokabüler Boyutu')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=preprocess_stats.index, y='model_size_mb', data=preprocess_stats)
        plt.title('Önişleme Tekniği - Ortalama Model Boyutu (MB)')
        plt.ylabel('Model Boyutu (MB)')
        
        plt.tight_layout()
        plt.savefig("preprocess_comparison.png")
        plt.close()
        
        # 5. Kombine parametreler
        print("\n5. En İyi Performans Gösteren Modeller:")
        best_models = stats_df.sort_values('vocabulary_size', ascending=False).head(3)
        print("En Büyük Vokabüler Boyutuna Sahip Modeller:")
        print(best_models[['preprocess', 'model_type', 'window', 'vector_size', 'vocabulary_size']])
        
        # Özet rapor hazırlama
        print("\n6. Özet Değerlendirme:")
        if model_type_stats.loc['skipgram', 'vocabulary_size'] > model_type_stats.loc['cbow', 'vocabulary_size']:
            print("- SkipGram modelleri genellikle daha büyük vokabüler boyutuna sahip")
        else:
            print("- CBOW modelleri genellikle daha büyük vokabüler boyutuna sahip")
        
        if window_stats.loc[4, 'vocabulary_size'] > window_stats.loc[2, 'vocabulary_size']:
            print("- Daha büyük pencere boyutu (4) daha fazla kelime ilişkisini yakalayabilmekte")
        else:
            print("- Daha küçük pencere boyutu (2) daha verimli çalışmakta")
        
        if vector_size_stats.loc[300, 'model_size_mb'] / vector_size_stats.loc[100, 'model_size_mb'] > 3:
            print("- Vektör boyutu arttıkça model boyutu orantısız olarak artmakta")
        else:
            print("- Vektör boyutu ile model boyutu arasında doğrusal bir ilişki var")
        
        return stats_df
    
    def generate_comprehensive_report(self, output_file="model_comparison_report.txt"):
        """
        Tüm model karşılaştırma sonuçlarını içeren kapsamlı bir rapor oluşturma
        """
        if not self.models or not self.model_stats:
            print("Karşılaştırma için yeterli veri bulunmuyor. Önce modelleri yükleyin ve karşılaştırın.")
            return
        
        # Rapor oluşturma
        with open(output_file, 'w', encoding='utf-8') as f:  # UTF-8 encoding ekledik
            f.write("# Word2Vec Model Karşılaştırma Raporu\n\n")
            
            # 1. Genel Model İstatistikleri
            f.write("## 1. Genel Model İstatistikleri\n\n")
            stats_df = pd.DataFrame.from_dict(self.model_stats, orient='index')
            f.write(f"{stats_df.to_string()}\n\n")
            
            # 2. Model Türü Karşılaştırması (CBOW vs SkipGram)
            f.write("## 2. Model Türü Karşılaştırması (CBOW vs SkipGram)\n\n")
            cbow_models = stats_df[stats_df['model_type'] == 'cbow']
            skipgram_models = stats_df[stats_df['model_type'] == 'skipgram']
            
            f.write(f"CBOW Modelleri Ortalama Vokabüler Boyutu: {cbow_models['vocabulary_size'].mean()}\n")
            f.write(f"SkipGram Modelleri Ortalama Vokabüler Boyutu: {skipgram_models['vocabulary_size'].mean()}\n\n")
            
            # 3. Pencere Boyutu Etkisi
            f.write("## 3. Pencere Boyutu Etkisi\n\n")
            window2_models = stats_df[stats_df['window'] == 2]
            window4_models = stats_df[stats_df['window'] == 4]
            
            f.write(f"Pencere Boyutu 2 Ortalama Vokabüler Boyutu: {window2_models['vocabulary_size'].mean()}\n")
            f.write(f"Pencere Boyutu 4 Ortalama Vokabüler Boyutu: {window4_models['vocabulary_size'].mean()}\n\n")
            
            # 4. Vektör Boyutu Etkisi
            f.write("## 4. Vektör Boyutu Etkisi\n\n")
            dim100_models = stats_df[stats_df['vector_size'] == 100]
            dim300_models = stats_df[stats_df['vector_size'] == 300]
            
            f.write(f"Vektör Boyutu 100 Ortalama Model Boyutu: {dim100_models['model_size_mb'].mean():.2f} MB\n")
            f.write(f"Vektör Boyutu 300 Ortalama Model Boyutu: {dim300_models['model_size_mb'].mean():.2f} MB\n\n")
            
            # 5. Önişleme Tekniği Etkisi
            f.write("## 5. Önişleme Tekniği Etkisi\n\n")
            lemma_models = stats_df[stats_df['preprocess'] == 'lemmatized']
            stem_models = stats_df[stats_df['preprocess'] == 'stemmed']
            
            f.write(f"Lemmatize Edilmiş Modeller Ortalama Vokabüler Boyutu: {lemma_models['vocabulary_size'].mean()}\n")
            f.write(f"Stem Edilmiş Modeller Ortalama Vokabüler Boyutu: {stem_models['vocabulary_size'].mean()}\n\n")
            
            # 6. Test Kelimeleri Benzerlik Sonuçları
            if self.similarity_results:
                f.write("## 6. Test Kelimeleri Benzerlik Sonuçları\n\n")
                
                # Burada test_words içindeki her kelime için karşılaştırma yapılıyor
                # self.test_words kullanmak yerine self.similarity_results içindeki kelimeleri kullanmalıyız
                
                # Her modelde hangi kelimelerin olduğunu kontrol et
                all_test_words = set()
                for model_results in self.similarity_results.values():
                    all_test_words.update(model_results.keys())
                
                # Her kelime için benzerlik sonuçlarını yaz
                for word in all_test_words:  # Düzeltilen kısım
                    f.write(f"### Kelime: '{word}'\n\n")
                    
                    for model_name, model_results in self.similarity_results.items():
                        f.write(f"#### {model_name}\n")
                        
                        if word in model_results and model_results[word]:
                            for similar_word, score in model_results[word]:
                                f.write(f"- {similar_word}: {score:.4f}\n")
                        else:
                            f.write("- Kelime bulunamadı veya benzerlik sonucu yok.\n")
                        
                        f.write("\n")
            
            # 7. Sonuç ve Öneriler
            f.write("## 7. Sonuç ve Öneriler\n\n")
            
            # En iyi model seçimi
            best_model_name = stats_df.sort_values(['vocabulary_size', 'model_size_mb'], ascending=[False, True]).index[0]
            f.write(f"### En İyi Model\n\n")
            f.write(f"İstatistiklere göre, en iyi model '{best_model_name}' olarak belirlenmiştir.\n\n")
            
            # Model türü karşılaştırması özeti
            f.write("### Model Türü Karşılaştırması\n\n")
            if skipgram_models['vocabulary_size'].mean() > cbow_models['vocabulary_size'].mean():
                f.write("SkipGram modelleri, genellikle CBOW modellerinden daha büyük vokabüler boyutuna sahiptir. ")
                f.write("SkipGram modeli, nadir kelimelerin temsili konusunda daha iyi performans gösterme eğilimindedir.\n\n")
            else:
                f.write("CBOW modelleri, genellikle SkipGram modellerinden daha büyük vokabüler boyutuna sahiptir. ")
                f.write("CBOW modeli, eğitim hızı ve yaygın kelimelerin temsili konusunda avantaj sağlayabilir.\n\n")
            
            # Pencere boyutu karşılaştırması özeti
            f.write("### Pencere Boyutu Karşılaştırması\n\n")
            if window4_models['vocabulary_size'].mean() > window2_models['vocabulary_size'].mean():
                f.write("Daha büyük pencere boyutu (4), daha küçük pencere boyutuna (2) göre daha fazla kelime ilişkisini yakalayabilmektedir. ")
                f.write("Bu, kelimelerin daha geniş bir bağlamda anlaşılmasını sağlar ancak eğitim süresi artar.\n\n")
            else:
                f.write("Daha küçük pencere boyutu (2), daha büyük pencere boyutuna (4) göre daha verimli çalışmaktadır. ")
                f.write("Bu, kelimelerin daha yakın bağlamdaki ilişkilerini yakalamak için yeterli olabilir ve eğitim süresi daha kısadır.\n\n")
            
            # Vektör boyutu karşılaştırması özeti
            f.write("### Vektör Boyutu Karşılaştırması\n\n")
            f.write("Daha büyük vektör boyutu (300), daha zengin kelime temsilleri sağlar ancak model boyutu ve eğitim süresi artar. ")
            f.write("Daha küçük vektör boyutu (100), daha verimli depolama ve hesaplama sağlar ancak bazı anlamsal nüanslar kaybedilebilir.\n\n")
            
            # Önişleme tekniği karşılaştırması özeti
            f.write("### Önişleme Tekniği Karşılaştırması\n\n")
            if lemma_models['vocabulary_size'].mean() > stem_models['vocabulary_size'].mean():
                f.write("Lemmatize edilmiş metinlerle eğitilen modeller, stem edilmiş metinlerle eğitilen modellerden daha büyük vokabüler boyutuna sahiptir. ")
                f.write("Lemmatization, kelimelerin anlamsal bütünlüğünü koruyarak daha zengin kelime temsilleri sağlar.\n\n")
            else:
                f.write("Stem edilmiş metinlerle eğitilen modeller, lemmatize edilmiş metinlerle eğitilen modellerden daha büyük vokabüler boyutuna sahiptir. ")
                f.write("Stemming, kelime köklerini daha agresif bir şekilde indirgeyerek daha kompakt bir vokabüler oluşturabilir.\n\n")
            
            f.write("### Film Öneri Sistemi İçin Öneriler\n\n")
            f.write("Film öneri sistemi için en uygun model, 'lemmatized_model_skipgram_window4_dim300' olarak değerlendirilmektedir. ")
            f.write("Bu model:\n")
            f.write("- Semantik ilişkileri en iyi yakalayan SkipGram mimarisine sahip\n")
            f.write("- Geniş bağlam penceresi (4) ile uzun mesafeli ilişkileri hesaba katıyor\n")
            f.write("- Yüksek boyutlu vektörler (300) ile zengin kelime temsilleri sağlıyor\n")
            f.write("- Lemmatization ile kelimelerin anlamsal bütünlüğünü koruyor\n\n")
            
            f.write("Bu model, film önerileri için TF-IDF tabanlı yaklaşıma göre daha anlamlı sonuçlar verebilir. ")
            f.write("Özellikle genre ve tema bazlı benzerlikler konusunda daha başarılı öneriler sunabilir.")
        
        print(f"Kapsamlı karşılaştırma raporu '{output_file}' dosyasına kaydedildi.")
        
        return output_file


class RecommendationSystem:
    """Film öneri sistemi"""
    def __init__(self, processed_data, use_lemmatized=True):
        """
        Film öneri sistemi başlatma
        
        Parameters:
        -----------
        processed_data: pd.DataFrame
            İşlenmiş film verilerini içeren DataFrame
        use_lemmatized: bool, default=True
            Lemmatize edilmiş metinleri kullanma (False ise stemmed metinler kullanılır)
        """
        self.movie_data = processed_data.copy()
        self.use_lemmatized = use_lemmatized
        
        # NaN değerlerini kontrol et ve temizle
        print("Veri temizleniyor ve NaN değerleri kontrol ediliyor...")
        
        # İşlenmiş metinlerin NaN kontrolü
        if 'lemmatized_text' in self.movie_data.columns:
            self.movie_data['lemmatized_text'] = self.movie_data['lemmatized_text'].fillna('')
        
        if 'stemmed_text' in self.movie_data.columns:
            self.movie_data['stemmed_text'] = self.movie_data['stemmed_text'].fillna('')
        
        # İşlenmiş metinleri kullanma
        if self.use_lemmatized:
            self.movie_data['processed_text'] = self.movie_data['lemmatized_text']
            print("Lemmatize edilmiş metinler kullanılıyor.")
        else:
            self.movie_data['processed_text'] = self.movie_data['stemmed_text']
            print("Stem edilmiş metinler kullanılıyor.")
        
        # Tüm işlenmiş metinlerin string olduğundan emin ol
        self.movie_data['processed_text'] = self.movie_data['processed_text'].astype(str)
        
        # Boş metinleri kontrol et
        empty_texts = sum(self.movie_data['processed_text'] == '')
        if empty_texts > 0:
            print(f"Uyarı: {empty_texts} adet boş işlenmiş metin var.")
        
        # TF-IDF vektörleştirme
        self.tfidf_matrix, self.vectorizer = self._vectorize_texts()
        
        # Benzerlik matrisini hesaplama
        self.similarity_matrix = self._calculate_similarity()
    
    def _vectorize_texts(self):
        """Metinleri TF-IDF ile vektörleştirme"""
        print("Metinler vektörleştiriliyor...")
        
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
        
        # Boş olmayan metinleri vektörleştir
        try:
            tfidf_matrix = vectorizer.fit_transform(self.movie_data['processed_text'])
            print(f"Vektörleştirme tamamlandı. Şekil: {tfidf_matrix.shape}")
        except ValueError as e:
            print(f"Vektörleştirme hatası: {e}")
            print("Geçersiz metinleri temizleyip tekrar deneniyor...")
            
            # Boş veya geçersiz metinleri filtrele
            valid_indices = [i for i, text in enumerate(self.movie_data['processed_text']) 
                            if isinstance(text, str) and len(text.strip()) > 0]
            
            # Geçerli metinleri içeren yeni bir DataFrame oluştur
            valid_data = self.movie_data.iloc[valid_indices].copy()
            self.movie_data = valid_data  # Ana veri setini güncelle
            
            # Tekrar dene
            tfidf_matrix = vectorizer.fit_transform(self.movie_data['processed_text'])
            print(f"Vektörleştirme başarılı. Şekil: {tfidf_matrix.shape}")
        
        return tfidf_matrix, vectorizer
    
    def _calculate_similarity(self):
        """Vektörler arasındaki benzerliği hesaplama"""
        print("Benzerlik matrisi hesaplanıyor...")
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"Benzerlik matrisi oluşturuldu. Şekil: {cosine_sim.shape}")
        return cosine_sim
    
    def get_recommendations(self, movie_title, top_n=5):
        """
        Verilen film için benzer filmleri önerme
        
        Parameters:
        -----------
        movie_title: str
            Öneri istenilen film adı
        top_n: int
            Önerilecek film sayısı
            
        Returns:
        --------
        pd.DataFrame
            Benzer filmlerin adları ve benzerlik skorları
        """
        # Film indeksini bulma
        try:
            idx = self.movie_data[self.movie_data['title'].str.lower() == movie_title.lower()].index[0]
        except:
            # Eğer tam eşleşme yoksa en yakın film adını bulma
            titles = self.movie_data['title'].str.lower().tolist()
            closest_titles = sorted(titles, key=lambda x: len(set(x.split()) & set(movie_title.lower().split())), reverse=True)
            
            if not closest_titles:
                return pd.DataFrame()
                
            idx = self.movie_data[self.movie_data['title'].str.lower() == closest_titles[0]].index[0]
            print(f"Tam eşleşme bulunamadı. Bunun yerine kullanılan film: '{self.movie_data.iloc[idx]['title']}'")
        
        # Benzerlik skorlarını alma
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Benzerlik skorlarına göre sıralama
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Kendisini çıkartma
        sim_scores = sim_scores[1:top_n+1]
        
        # Film indekslerini alma
        movie_indices = [i[0] for i in sim_scores]
        
        # Öneri listesini oluşturma
        recommendations = pd.DataFrame({
            'title': self.movie_data.iloc[movie_indices]['title'].values,
            'similarity_score': [i[1] for i in sim_scores]
        })
        
        return recommendations
    
    def visualize_recommendations(self, recommendations):
        """
        Önerileri görselleştirme
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x='similarity_score', y='title', data=recommendations)
        plt.title('Film Önerileri ve Benzerlik Skorları')
        plt.xlabel('Benzerlik Skoru')
        plt.ylabel('Film Adı')
        plt.tight_layout()
        plt.savefig("recommendations.png")
        plt.close()
    
    def save_models(self, output_dir="models"):
        """Modelleri kaydetme"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # TF-IDF vektörlüştürücüyü kaydetme
        joblib.dump(self.vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    
        # Benzerlik matrisini kaydetme
        joblib.dump(self.similarity_matrix, os.path.join(output_dir, "similarity_matrix.pkl"))
    
        # İşlenmiş veriyi kaydetme - bu ayrı bir klasöre kaydedilmeli
        processed_data_dir = "processed_data"
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        self.movie_data.to_csv(os.path.join(processed_data_dir, "recommendation_data.csv"), index=False)
    
        print(f"Modeller '{output_dir}' dizinine kaydedildi.")
        print(f"İşlenmiş veri '{processed_data_dir}' dizinine kaydedildi.")
    
    @classmethod
    def load_models(cls, model_dir="models", data_path=None):
        """Kaydedilmiş modelleri yükleme"""
        # Veriyi yükleme
        if data_path:
            movie_data = pd.read_csv(data_path)
        else:
            movie_data = pd.read_csv(os.path.join("processed_data", "recommendation_data.csv"))
        
        # Vektörleştirici ve benzerlik matrisini yükleme
        vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        similarity_matrix = joblib.load(os.path.join(model_dir, "similarity_matrix.pkl"))
        
        # Recommendation System nesnesi oluşturma
        recommender = cls(movie_data)
        
        # Yüklenen modelleri atama
        recommender.vectorizer = vectorizer
        recommender.similarity_matrix = similarity_matrix
        
        print("Öneri sistemi modelleri başarıyla yüklendi.")
        return recommender
    
    def compare_w2v_tfidf_recommendations(self, movie_title, w2v_model, top_n=5):
        """
        Word2Vec ve TF-IDF tabanlı film önerilerini karşılaştırma
        
        Parameters:
        -----------
        movie_title: str
            Öneri istenilen film adı
        w2v_model: Word2Vec
            Karşılaştırma için kullanılacak Word2Vec modeli
        top_n: int
            Önerilecek film sayısı
            
        Returns:
        --------
        tuple
            (tfidf_recommendations, w2v_recommendations) içeren tuple
        """
        # TF-IDF tabanlı öneriler
        tfidf_recommendations = self.get_recommendations(movie_title, top_n)
        
        # Word2Vec tabanlı öneriler
        # Film adını kelimelerine ayırma
        movie_words = movie_title.lower().split()
        
        # Vektör hesaplama
        try:
            # Modelde bulunan film adı kelimelerini al
            valid_words = [word for word in movie_words if word in w2v_model.wv]
            
            if not valid_words:
                print(f"'{movie_title}' filminin kelimeleri Word2Vec modelinde bulunamadı.")
                return tfidf_recommendations, pd.DataFrame()
            
            # Kelimelerin vektörlerinin ortalamasını al
            movie_vector = np.mean([w2v_model.wv[word] for word in valid_words], axis=0)
            
            # Tüm filmler için vektörler hesapla
            all_movie_vectors = []
            for idx, row in self.movie_data.iterrows():
                title_words = row['title'].lower().split()
                valid_title_words = [word for word in title_words if word in w2v_model.wv]
                if valid_title_words:
                    vector = np.mean([w2v_model.wv[word] for word in valid_title_words], axis=0)
                    all_movie_vectors.append((idx, vector))
            
            # Benzerlik hesapla
            similarities = []
            for idx, vector in all_movie_vectors:
                sim = cosine_similarity([movie_vector], [vector])[0][0]
                similarities.append((idx, sim))
            
            # Benzerlik skorlarına göre sırala
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Kendisini çıkar ve top_n kadar al
            target_idx = self.movie_data[self.movie_data['title'].str.lower() == movie_title.lower()].index
            if len(target_idx) > 0:
                target_idx = target_idx[0]
                similarities = [(i, s) for i, s in similarities if i != target_idx]
            
            similarities = similarities[:top_n]
            
            # Öneri listesini oluştur
            movie_indices = [i for i, _ in similarities]
            w2v_recommendations = pd.DataFrame({
                'title': self.movie_data.iloc[movie_indices]['title'].values,
                'similarity_score': [s for _, s in similarities]
            })
            
            return tfidf_recommendations, w2v_recommendations
            
        except Exception as e:
            print(f"Word2Vec tabanlı öneri hesaplanırken hata: {e}")
            return tfidf_recommendations, pd.DataFrame()
    
    def visualize_recommendation_comparison(self, tfidf_recommendations, w2v_recommendations, movie_title):
        """
        TF-IDF ve Word2Vec tabanlı film önerilerini karşılaştırmalı olarak görselleştirme
        """
        plt.figure(figsize=(12, 8))
        
        # TF-IDF önerileri
        plt.subplot(1, 2, 1)
        sns.barplot(x='similarity_score', y='title', data=tfidf_recommendations)
        plt.title('TF-IDF Tabanlı Film Önerileri')
        plt.xlabel('Benzerlik Skoru')
        plt.ylabel('Film Adı')
        
        # Word2Vec önerileri
        plt.subplot(1, 2, 2)
        if not w2v_recommendations.empty:
            sns.barplot(x='similarity_score', y='title', data=w2v_recommendations)
            plt.title('Word2Vec Tabanlı Film Önerileri')
            plt.xlabel('Benzerlik Skoru')
            plt.ylabel('Film Adı')
        else:
            plt.text(0.5, 0.5, "Word2Vec önerileri hesaplanamadı", ha='center', va='center')
            plt.title('Word2Vec Tabanlı Film Önerileri')
            plt.axis('off')
        
        plt.suptitle(f"'{movie_title}' İçin Film Önerisi Karşılaştırması", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"recommendation_comparison_{movie_title.replace(' ', '_')}.png")
        plt.close()
        
        print(f"'{movie_title}' için öneri karşılaştırması görselleştirildi.")


def prepare_data_for_word2vec(processed_data):
    """Word2Vec eğitimi için verileri hazırlama"""
    lemmatized_corpus = []
    stemmed_corpus = []
    
    print("Word2Vec eğitimi için veriler hazırlanıyor...")
    
    # Her film için kelime listesi oluşturma
    for _, row in processed_data.iterrows():
        # NaN değerleri kontrol et
        lemma_text = str(row['lemmatized_text']) if pd.notna(row['lemmatized_text']) else ""
        stem_text = str(row['stemmed_text']) if pd.notna(row['stemmed_text']) else ""
        
        # Lemmatized metni tokenize etme
        lemma_tokens = lemma_text.split()
        if lemma_tokens:
            lemmatized_corpus.append(lemma_tokens)
        
        # Stemmed metni tokenize etme
        stem_tokens = stem_text.split()
        if stem_tokens:
            stemmed_corpus.append(stem_tokens)
    
    print(f"Toplam {len(lemmatized_corpus)} lemmatized ve {len(stemmed_corpus)} stemmed metin hazırlandı.")
    
    return lemmatized_corpus, stemmed_corpus


def test_word2vec_models(model_dir="models"):
    """Eğitilmiş Word2Vec modellerini test etme"""
    # 3 modeli yükle
    try:
        print("Word2Vec modellerini test etme...")
        model_1 = Word2Vec.load(os.path.join(model_dir, "lemmatized_model_cbow_window2_dim100.model"))
        model_2 = Word2Vec.load(os.path.join(model_dir, "stemmed_model_skipgram_window4_dim100.model"))
        model_3 = Word2Vec.load(os.path.join(model_dir, "lemmatized_model_skipgram_window2_dim300.model"))
        
        # Benzer kelimeleri bulma fonksiyonu
        def print_similar_words(model, model_name):
            try:
                # Önce 'movie' kelimesiyle deneyelim
                similarity = model.wv.most_similar('movie', topn=3)
                print(f"\n{model_name} Modeli - 'movie' ile En Benzer 3 Kelime:")
                for word, score in similarity:
                    print(f"Kelime: {word}, Benzerlik Skoru: {score}")
            except KeyError:
                print(f"\n{model_name} Modeli için 'movie' kelimesi bulunamadı.")
                # Modelde bulunan başka bir kelimeyi deneyelim
                if len(model.wv.index_to_key) > 0:
                    test_word = model.wv.index_to_key[0]
                    print(f"Model içinde bulunan kelimeler: {model.wv.index_to_key[:5]}")
                    similarity = model.wv.most_similar(test_word, topn=3)
                    print(f"\n{model_name} Modeli - '{test_word}' ile En Benzer 3 Kelime:")
                    for word, score in similarity:
                        print(f"Kelime: {word}, Benzerlik Skoru: {score}")
        
        # 3 model için benzer kelimeleri yazdır
        print_similar_words(model_1, "Lemmatized CBOW Window 2 Dim 100")
        print_similar_words(model_2, "Stemmed Skipgram Window 4 Dim 100")
        print_similar_words(model_3, "Lemmatized Skipgram Window 2 Dim 300")
        
        return True
    except Exception as e:
        print(f"Word2Vec modelleri test edilirken bir hata oluştu: {e}")
        return False


def compare_tfidf_word2vec(text, tfidf_analyzer, word2vec_model, word, top_n=5):
    """TF-IDF ve Word2Vec benzerliklerini karşılaştırma"""
    print(f"'{word}' kelimesi için TF-IDF ve Word2Vec karşılaştırması:\n")
    
    # TF-IDF sonuçları
    print("TF-IDF Benzer Kelimeler:")
    tfidf_similar = tfidf_analyzer.find_similar_words(word, top_n)
    if tfidf_similar:
        for w, score in tfidf_similar:
            print(f"  {w}: {score:.4f}")
    else:
        print("  TF-IDF modeli için sonuç bulunamadı.")
    
    # Word2Vec sonuçları
    print("\nWord2Vec Benzer Kelimeler:")
    try:
        w2v_similar = word2vec_model.wv.most_similar(word, topn=top_n)
        for w, score in w2v_similar:
            print(f"  {w}: {score:.4f}")
    except KeyError:
        print(f"  '{word}' kelimesi Word2Vec modelinde bulunamadı.")
    
    # Görselleştirme
    plt.figure(figsize=(12, 6))
    
    # TF-IDF grafiği
    if tfidf_similar:
        plt.subplot(1, 2, 1)
        tfidf_words = [w for w, _ in tfidf_similar]
        tfidf_scores = [s for _, s in tfidf_similar]
        sns.barplot(x=tfidf_scores, y=tfidf_words)
        plt.title(f"TF-IDF: '{word}' Benzer Kelimeler")
        plt.xlabel('Benzerlik Skoru')
    
    # Word2Vec grafiği
    try:
        plt.subplot(1, 2, 2)
        w2v_words = [w for w, _ in w2v_similar]
        w2v_scores = [s for _, s in w2v_similar]
        sns.barplot(x=w2v_scores, y=w2v_words)
        plt.title(f"Word2Vec: '{word}' Benzer Kelimeler")
        plt.xlabel('Benzerlik Skoru')
    except:
        pass
    
    plt.tight_layout()
    plt.savefig(f"comparison_{word}.png")
    plt.close()


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
    
    return kelimeler, siralar, frekanslar


def main():
    """
    Ana işlevi çalıştırma - Film öneri sistemi model eğitimi ve karşılaştırması
    """
    print("Film ve Dizi Öneri Sistemi Model Eğitimi")
    print("=" * 70)
    
    # 1. İşlenmiş Verileri Yükleme
    print("\n1. İşlenmiş Verileri Yükleme")
    print("-" * 70)
    
    # Kullanıcıdan veri yolu seçmesini isteme
    data_dir = input("İşlenmiş verilerin bulunduğu dizin (varsayılan: processed_data): ") or "processed_data"
    
    try:
        # Ana işlenmiş verileri yükleme
        processed_data_path = os.path.join(data_dir, "processed_movies.csv")
        if not os.path.exists(processed_data_path):
            print(f"Hata: '{processed_data_path}' dosyası bulunamadı.")
            return
            
        processed_data = pd.read_csv(processed_data_path)
        print(f"İşlenmiş veriler yüklendi: {len(processed_data)} film")
        
        # Zipf grafiklerini model eğitiminde de kullanacaksak
        if not os.path.exists("zipf_graphs"):
            os.makedirs("zipf_graphs")
            
            print("Zipf yasası analizleri yapılıyor...")
            # Tüm metin verilerini birleştir
            all_lemmatized_text = ' '.join(processed_data['lemmatized_text'].fillna('').astype(str))
            all_stemmed_text = ' '.join(processed_data['stemmed_text'].fillna('').astype(str))
            
            # Zipf analizleri
            lemma_words, lemma_ranks, lemma_freqs = zipf_analizi_ciz(all_lemmatized_text, "Lemmatize Edilmiş Veri")
            stem_words, stem_ranks, stem_freqs = zipf_analizi_ciz(all_stemmed_text, "Stem Edilmiş Veri")
            
            # Zipf analizleri karşılaştırma grafiği
            plt.figure(figsize=(12, 8))
            plt.loglog(lemma_ranks[:1000], lemma_freqs[:1000], 'g.', alpha=0.5, label='Lemmatize Edilmiş')
            plt.loglog(stem_ranks[:1000], stem_freqs[:1000], 'r.', alpha=0.5, label='Stem Edilmiş')
            
            # Teorik Zipf eğrisi
            plt.loglog(lemma_ranks[:1000], [lemma_freqs[0]/r for r in lemma_ranks[:1000]], 'k-', 
                      label='Zipf Yasası (1/rank)')
            
            plt.xlabel('Sıra (log ölçeği)')
            plt.ylabel('Frekans (log ölçeği)')
            plt.title('Zipf Yasası Karşılaştırması: Lemmatize vs. Stem Edilmiş Veri')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join("zipf_graphs", "zipf_comparison_processed.png"))
            plt.close()
            
            # Kelime dağarcığı boyutları karşılaştırma
            print("\nKelime dağarcığı karşılaştırması:")
            print(f"Lemmatize edilmiş kelime dağarcığı boyutu: {len(lemma_words)}")
            print(f"Stem edilmiş kelime dağarcığı boyutu: {len(stem_words)}")
            print(f"Kelime dağarcığı değişimi: {(len(lemma_words) - len(stem_words))/len(lemma_words)*100:.2f}%")
        
    except Exception as e:
        print(f"Hata: İşlenmiş veriler yüklenemedi - {e}")
        return
    
    # 2. TF-IDF Analizi
    print("\n2. TF-IDF Analizi")
    print("-" * 70)
    
    # Lemmatize/Stem seçimi
    lemma_or_stem = input("Lemmatize edilmiş metin kullanmak ister misiniz? (e/h, varsayılan: e): ")
    use_lemmatized = lemma_or_stem.lower() != 'h'
    
    # TF-IDF analizi
    if use_lemmatized:
        texts = processed_data['lemmatized_text'].fillna('').astype(str).tolist()
        print("Lemmatize edilmiş metinler kullanılıyor...")
    else:
        texts = processed_data['stemmed_text'].fillna('').astype(str).tolist()
        print("Stem edilmiş metinler kullanılıyor...")
    
    # TF-IDF analizör oluşturma
    tfidf_analyzer = TfidfAnalyzer(texts)
    
    # Birkaç örnek kelime için benzer kelimeler
    example_words = ['movie', 'action', 'love', 'hero', 'villain']
    print("\nTF-IDF ile Kelime Benzerliği Analizi:")
    for word in example_words:
        similar_words = tfidf_analyzer.find_similar_words(word)
        if similar_words:
            print(f"\n'{word}' kelimesine benzer kelimeler:")
            for w, score in similar_words:
                print(f"  {w}: {score:.4f}")
            
            # Visualize
            tfidf_analyzer.visualize_word_similarities(word, similar_words)
    
    # TF-IDF modelini kaydetme
    tfidf_analyzer.save_model(output_dir="models")
    
    # 3. Öneri Sistemi Modeli Eğitimi
    print("\n3. Öneri Sistemi Modeli Eğitimi")
    print("-" * 70)
    
    # Öneri sistemi oluşturma
    recommender = RecommendationSystem(processed_data, use_lemmatized=use_lemmatized)
    
    # Modelleri kaydetme
    output_dir = "models"
    recommender.save_models(output_dir=output_dir)
    
    # 4. Word2Vec Modelleri Eğitimi
    print("\n4. Word2Vec Modelleri Eğitimi")
    print("-" * 70)
    
    train_w2v = input("Word2Vec modellerini eğitmek ister misiniz? (e/h, varsayılan: h): ")
    if train_w2v.lower() == 'e':
        # Word2Vec eğitimi için verileri hazırlama
        lemmatized_corpus, stemmed_corpus = prepare_data_for_word2vec(processed_data)
        
        # Word2Vec eğitici oluşturma
        w2v_trainer = Word2VecTrainer()
        
        # Modelleri eğitme
        w2v_models = w2v_trainer.train_all_models(lemmatized_corpus, stemmed_corpus, output_dir=output_dir)
        
        # Eğitim istatistiklerini görselleştirme
        w2v_trainer.visualize_training_stats()
        
        # Modelleri test etme
        test_w2v = input("Eğitilen Word2Vec modellerini test etmek ister misiniz? (e/h, varsayılan: h): ")
        if test_w2v.lower() == 'e':
            test_word2vec_models(model_dir=output_dir)
        
        # Model Karşılaştırıcı
        compare_models = input("Modelleri detaylı olarak karşılaştırmak ister misiniz? (e/h, varsayılan: h): ")
        if compare_models.lower() == 'e':
            # Model karşılaştırıcı oluşturma
            print("\nWord2Vec Modelleri Karşılaştırılıyor...")
            comparator = ModelComparator(models_dir=output_dir)
            
            # Modelleri yükleme
            comparator.load_models()
            
            # Kelimeleri karşılaştırma
            test_words = input("Test edilecek kelimeleri virgülle ayırarak girin (varsayılan: movie,love,action,hero,villain): ")
            if test_words.strip():
                test_words_list = [word.strip() for word in test_words.split(",")]
            else:
                test_words_list = ["movie", "love", "action", "hero", "villain"]
                
            comparator.compare_similar_words(test_words=test_words_list, top_n=5)
            
            # Kelime vektörlerini görselleştirme
            for word in test_words_list:
                for model_name in list(comparator.models.keys())[:3]:  # İlk 3 model için
                    comparator.visualize_word_similarities(word, top_n=5)
            
            # Parametre etki analizi
            print("\nModel parametrelerinin etkisi analiz ediliyor...")
            stats_df = comparator.compare_model_parameters()
            
            # Kapsamlı rapor oluşturma
            print("\nKapsamlı karşılaştırma raporu oluşturuluyor...")
            report_file = comparator.generate_comprehensive_report()
            
            print(f"\nModel karşılaştırma raporu '{report_file}' dosyasına kaydedildi.")
        
        # TF-IDF ve Word2Vec karşılaştırması
        compare_tf_w2v = input("TF-IDF ve Word2Vec sonuçlarını karşılaştırmak ister misiniz? (e/h, varsayılan: h): ")
        if compare_tf_w2v.lower() == 'e':
            # Bir Word2Vec modeli seç
            model_name = "lemmatized_model_skipgram_window2_dim300.model"
            model_path = os.path.join(output_dir, model_name)
            
            if os.path.exists(model_path):
                model = Word2Vec.load(model_path)
                print(f"'{model_name}' modeli yüklendi.")
                
                while True:
                    word = input("\nKarşılaştırma için bir kelime girin (çıkmak için 'q'): ")
                    if word.lower() == 'q':
                        break
                    compare_tfidf_word2vec(texts, tfidf_analyzer, model, word)
            else:
                print(f"Hata: '{model_path}' dosyası bulunamadı.")
    else:
        # Önceden eğitilmiş modelleri yükleme
        print("Mevcut Word2Vec modelleri yükleniyor...")
        # Modellerin varlığını kontrol et
        model_path = os.path.join(output_dir, "lemmatized_model_skipgram_window4_dim300.model")
        if os.path.exists(model_path):
            # Model karşılaştırıcı
            comparator = ModelComparator(models_dir=output_dir)
            comparator.load_models()
            
            print("Modeller başarıyla yüklendi. Detaylı karşılaştırma yapmak ister misiniz? (e/h, varsayılan: h): ")
            do_compare = input()
            
            if do_compare.lower() == 'e':
                # Bazı test kelimeleri için benzerlik karşılaştırması
                test_words = ["movie", "love", "action", "hero", "villain"]
                comparator.compare_similar_words(test_words=test_words, top_n=5)
                
                # Parametre etki analizi
                stats_df = comparator.compare_model_parameters()
                
                # Rapor oluşturma
                report_file = comparator.generate_comprehensive_report()
                print(f"\nModel karşılaştırma raporu '{report_file}' dosyasına kaydedildi.")
        else:
            print("Hata: Word2Vec modelleri bulunamadı. Önce modelleri eğitmeniz gerekiyor.")
    
    # 5. Film Öneri Sistemi Kullanımı
    print("\n5. Film Öneri Sistemi Hazır")
    print("-" * 70)
    
    while True:
        search_title = input("\nHangi film için öneri istiyorsunuz? (Çıkmak için 'q'): ")
        
        if search_title.lower() == 'q':
            break
        
        top_n = int(input("Kaç film önerisi istiyorsunuz? (varsayılan: 5): ") or "5")
        
        recommendations = recommender.get_recommendations(search_title, top_n=top_n)
        
        if recommendations.empty:
            print(f"'{search_title}' için öneri bulunamadı.")
        else:
            print(f"\n'{search_title}' filmine benzer {top_n} film önerisi:")
            print(recommendations)
            
            # Görselleştirme
            viz_choice = input("Önerileri görselleştirmek ister misiniz? (e/h): ")
            if viz_choice.lower() == 'e':
                recommender.visualize_recommendations(recommendations)
            
            # TF-IDF ve Word2Vec karşılaştırmalı öneriler
            w2v_compare = input("TF-IDF ve Word2Vec tabanlı önerileri karşılaştırmak ister misiniz? (e/h): ")
            if w2v_compare.lower() == 'e':
                # Word2Vec modelini yükleme
                model_path = os.path.join(output_dir, "lemmatized_model_skipgram_window4_dim300.model")
                if os.path.exists(model_path):
                    w2v_model = Word2Vec.load(model_path)
                    
                    # Önerileri karşılaştırma
                    tfidf_recs, w2v_recs = recommender.compare_w2v_tfidf_recommendations(search_title, w2v_model, top_n=top_n)
                    
                    # Görselleştirme
                    recommender.visualize_recommendation_comparison(tfidf_recs, w2v_recs, search_title)
                else:
                    print("Hata: Word2Vec modeli bulunamadı. Önce modelleri eğitmeniz gerekiyor.")
    
    print("\nFilm Öneri Sistemi Model Eğitimi ve Karşılaştırması tamamlandı.")
    print("Raporlar, grafikler ve model dosyaları ilgili dizinlere kaydedildi.")


if __name__ == "__main__":
    main()