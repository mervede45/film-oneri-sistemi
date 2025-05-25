import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import joblib
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class SimilarityEvaluator:
    """
    Metin benzerliği hesaplama ve değerlendirme sistemi
    TF-IDF ve Word2Vec modellerini karşılaştırır
    """
    
    def __init__(self, processed_data_path="processed_data", models_path="models"):
        self.processed_data_path = processed_data_path
        self.models_path = models_path
        self.movie_data = None
        self.tfidf_lemmatized = None
        self.tfidf_stemmed = None
        self.tfidf_vectorizer = None
        self.word2vec_models = {}
        self.similarity_results = {}
        self.evaluation_results = {}
        
        # Model adları
        self.model_names = []
        
        # Sonuçları saklamak için
        self.top5_results = {}
        self.semantic_scores = {}
        self.jaccard_matrix = None
        
    def load_data_and_models(self):
        """Verileri ve modelleri yükleme"""
        print("Veriler ve modeller yükleniyor...")
        
        # İşlenmiş verileri yükleme
        try:
            self.movie_data = pd.read_csv(os.path.join(self.processed_data_path, "processed_movies.csv"))
            print(f"Film verileri yüklendi: {len(self.movie_data)} film")
        except Exception as e:
            print(f"Film verileri yüklenemedi: {e}")
            return False
        
        # TF-IDF vektörlerini yükleme
        try:
            # TF-IDF CSV dosyalarının mevcut olup olmadığını kontrol et
            lemma_csv = os.path.join(self.processed_data_path, "tfidf_lemmatized.csv")
            stem_csv = os.path.join(self.processed_data_path, "tfidf_stemmed.csv")
            
            if os.path.exists(lemma_csv) and os.path.exists(stem_csv):
                self.tfidf_lemmatized = pd.read_csv(lemma_csv, index_col=0)
                self.tfidf_stemmed = pd.read_csv(stem_csv, index_col=0)
                print("TF-IDF vektörleri CSV'den yüklendi")
            else:
                print("TF-IDF CSV dosyaları bulunamadı, yeniden oluşturuluyor...")
                self._create_tfidf_vectors()
                
        except Exception as e:
            print(f"TF-IDF vektörleri yüklenirken hata: {e}")
            print("Yeniden oluşturuluyor...")
            self._create_tfidf_vectors()
        
        # TF-IDF vektörleştiriciyi yükleme
        try:
            self.tfidf_vectorizer = joblib.load(os.path.join(self.models_path, "tfidf_vectorizer.pkl"))
            print("TF-IDF vektörleştirici yüklendi")
        except Exception as e:
            print(f"TF-IDF vektörleştirici yüklenemedi: {e}")
        
        # Word2Vec modellerini yükleme
        self._load_word2vec_models()
        
        # Model adlarını oluşturma
        self.model_names = ["tfidf_lemmatized", "tfidf_stemmed"] + list(self.word2vec_models.keys())
        print(f"Toplam {len(self.model_names)} model yüklendi")
        
        return True
    
    def _create_tfidf_vectors(self):
        """TF-IDF vektörlerini yeniden oluşturma"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            print("TF-IDF vektörleri oluşturuluyor...")
            
            # Lemmatized metinler için TF-IDF
            lemma_texts = self.movie_data['lemmatized_text'].fillna('').astype(str).tolist()
            vectorizer_lemma = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
            tfidf_matrix_lemma = vectorizer_lemma.fit_transform(lemma_texts)
            
            self.tfidf_lemmatized = pd.DataFrame(
                tfidf_matrix_lemma.toarray(),
                columns=vectorizer_lemma.get_feature_names_out(),
                index=self.movie_data['title']
            )
            
            # Stemmed metinler için TF-IDF  
            stem_texts = self.movie_data['stemmed_text'].fillna('').astype(str).tolist()
            vectorizer_stem = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
            tfidf_matrix_stem = vectorizer_stem.fit_transform(stem_texts)
            
            self.tfidf_stemmed = pd.DataFrame(
                tfidf_matrix_stem.toarray(),
                columns=vectorizer_stem.get_feature_names_out(),
                index=self.movie_data['title']
            )
            
            # CSV'ye kaydetme
            self.tfidf_lemmatized.to_csv(os.path.join(self.processed_data_path, "tfidf_lemmatized.csv"))
            self.tfidf_stemmed.to_csv(os.path.join(self.processed_data_path, "tfidf_stemmed.csv"))
            
            print("TF-IDF vektörleri başarıyla oluşturuldu ve kaydedildi")
            
        except Exception as e:
            print(f"TF-IDF vektörleri oluşturulamadı: {e}")
            return False
    
    def _load_word2vec_models(self):
        """Word2Vec modellerini yükleme"""
        print("Word2Vec modelleri yükleniyor...")
        
        # Model dosyalarını bulma
        model_files = [f for f in os.listdir(self.models_path) if f.endswith('.model')]
        
        for model_file in model_files:
            try:
                model_path = os.path.join(self.models_path, model_file)
                model = Word2Vec.load(model_path)
                model_name = model_file.replace('.model', '')
                self.word2vec_models[model_name] = model
                print(f"Yüklendi: {model_name}")
            except Exception as e:
                print(f"Hata: {model_file} yüklenemedi - {e}")
    
    def select_input_text(self, movie_title=None, movie_index=None):
        """Giriş metnini seçme"""
        if movie_title:
            # Film adıyla arama
            matching_movies = self.movie_data[self.movie_data['title'].str.contains(movie_title, case=False, na=False)]
            if len(matching_movies) > 0:
                selected_movie = matching_movies.iloc[0]
                print(f"Seçilen film: {selected_movie['title']}")
                return selected_movie
            else:
                print(f"'{movie_title}' adında film bulunamadı.")
                return None
        
        elif movie_index is not None:
            if 0 <= movie_index < len(self.movie_data):
                selected_movie = self.movie_data.iloc[movie_index]
                print(f"Seçilen film: {selected_movie['title']}")
                return selected_movie
            else:
                print(f"Geçersiz indeks: {movie_index}")
                return None
        
        else:
            # Rastgele film seçme
            random_index = np.random.randint(0, len(self.movie_data))
            selected_movie = self.movie_data.iloc[random_index]
            print(f"Rastgele seçilen film: {selected_movie['title']}")
            return selected_movie
    
    def calculate_tfidf_similarity(self, input_movie, model_type="lemmatized"):
        """TF-IDF benzerliği hesaplama"""
        if model_type == "lemmatized":
            tfidf_data = self.tfidf_lemmatized
        else:
            tfidf_data = self.tfidf_stemmed
        
        # Giriş filminin TF-IDF vektörünü bulma
        input_title = input_movie['title']
        
        if input_title not in tfidf_data.index:
            print(f"'{input_title}' filmi TF-IDF vektörlerinde bulunamadı.")
            return []
        
        # Giriş vektörü
        input_vector = tfidf_data.loc[input_title].values.reshape(1, -1)
        
        # Tüm filmlerle benzerlik hesaplama
        similarities = []
        for title in tfidf_data.index:
            if title != input_title:  # Kendisini hariç tutma
                film_vector = tfidf_data.loc[title].values.reshape(1, -1)
                
                # Vektör boyutlarını kontrol et ve uyumlu hale getir
                if input_vector.shape[1] == film_vector.shape[1]:
                    sim_score = cosine_similarity(input_vector, film_vector)[0][0]
                    similarities.append((title, sim_score))
                else:
                    print(f"Vektör boyutu uyumsuz: {title}")
        
        # Benzerlik skoruna göre sıralama ve ilk 5'i alma
        similarities.sort(key=lambda x: x[1], reverse=True)
        top5 = similarities[:5]
        
        return top5
    
    def calculate_word2vec_similarity(self, input_movie, model_name):
        """Word2Vec benzerliği hesaplama"""
        if model_name not in self.word2vec_models:
            print(f"'{model_name}' modeli bulunamadı.")
            return []
        
        model = self.word2vec_models[model_name]
        
        # Giriş metnini tokenize etme
        if "lemmatized" in model_name:
            input_text = str(input_movie['lemmatized_text'])
        else:
            input_text = str(input_movie['stemmed_text'])
        
        input_tokens = input_text.split()
        
        # Giriş metni için ortalama vektör hesaplama
        input_vectors = []
        for token in input_tokens:
            if token in model.wv:
                input_vectors.append(model.wv[token])
        
        if not input_vectors:
            print(f"Giriş metni kelimeleri '{model_name}' modelinde bulunamadı.")
            return []
        
        input_avg_vector = np.mean(input_vectors, axis=0).reshape(1, -1)
        
        # Tüm filmlerle benzerlik hesaplama
        similarities = []
        for idx, row in self.movie_data.iterrows():
            if row['title'] == input_movie['title']:
                continue  # Kendisini hariç tutma
            
            # Film metnini tokenize etme
            if "lemmatized" in model_name:
                film_text = str(row['lemmatized_text'])
            else:
                film_text = str(row['stemmed_text'])
            
            film_tokens = film_text.split()
            
            # Film için ortalama vektör hesaplama
            film_vectors = []
            for token in film_tokens:
                if token in model.wv:
                    film_vectors.append(model.wv[token])
            
            if film_vectors:
                film_avg_vector = np.mean(film_vectors, axis=0).reshape(1, -1)
                
                # Vektör boyutlarını kontrol et
                if input_avg_vector.shape[1] == film_avg_vector.shape[1]:
                    sim_score = cosine_similarity(input_avg_vector, film_avg_vector)[0][0]
                    similarities.append((row['title'], sim_score))
        
        # Benzerlik skoruna göre sıralama ve ilk 5'i alma
        similarities.sort(key=lambda x: x[1], reverse=True)
        top5 = similarities[:5]
        
        return top5
    
    def calculate_all_similarities(self, input_movie):
        """Tüm modeller için benzerlik hesaplama"""
        print(f"\n'{input_movie['title']}' filmi için benzerlik hesaplamaları yapılıyor...")
        
        self.top5_results = {}
        
        # TF-IDF modelleri
        print("\nTF-IDF Benzerlik Hesaplamaları:")
        try:
            tfidf_lemma_results = self.calculate_tfidf_similarity(input_movie, "lemmatized")
            self.top5_results["tfidf_lemmatized"] = tfidf_lemma_results
            print(f"TF-IDF Lemmatized: {len(tfidf_lemma_results)} sonuç")
        except Exception as e:
            print(f"TF-IDF Lemmatized hatası: {e}")
            self.top5_results["tfidf_lemmatized"] = []
        
        try:
            tfidf_stem_results = self.calculate_tfidf_similarity(input_movie, "stemmed")
            self.top5_results["tfidf_stemmed"] = tfidf_stem_results
            print(f"TF-IDF Stemmed: {len(tfidf_stem_results)} sonuç")
        except Exception as e:
            print(f"TF-IDF Stemmed hatası: {e}")
            self.top5_results["tfidf_stemmed"] = []
        
        # Word2Vec modelleri
        print("\nWord2Vec Benzerlik Hesaplamaları:")
        for model_name in self.word2vec_models.keys():
            try:
                w2v_results = self.calculate_word2vec_similarity(input_movie, model_name)
                self.top5_results[model_name] = w2v_results
                print(f"{model_name}: {len(w2v_results)} sonuç")
            except Exception as e:
                print(f"{model_name} hatası: {e}")
                self.top5_results[model_name] = []
        
        return self.top5_results
    
    def display_results(self):
        """Sonuçları görüntüleme"""
        print("\n" + "="*80)
        print("BENZERLIK HESAPLAMA SONUÇLARI")
        print("="*80)
        
        for model_name, results in self.top5_results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 50)
            if results:
                for i, (title, score) in enumerate(results, 1):
                    print(f"{i}. {title} (Skor: {score:.4f})")
            else:
                print("Sonuç bulunamadı")
    
    def semantic_evaluation(self):
        """Anlamsal değerlendirme (Manuel puanlama için)"""
        print("\n" + "="*80)
        print("ANLAMSAL DEĞERLENDİRME")
        print("="*80)
        print("Her modelin önerdiği filmler için 1-5 arası puan veriniz:")
        print("1: Çok alakasız, 2: Kısmen ilgili, 3: Ortalama benzer, 4: Anlamlı benzer, 5: Çok güçlü benzerlik")
        
        self.semantic_scores = {}
        
        for model_name, results in self.top5_results.items():
            print(f"\n{model_name} Modeli Değerlendirmesi:")
            scores = []
            
            if results:
                for i, (title, similarity_score) in enumerate(results, 1):
                    while True:
                        try:
                            score = input(f"{i}. {title} (Benzerlik: {similarity_score:.4f}) - Puanınız (1-5): ")
                            score = int(score)
                            if 1 <= score <= 5:
                                scores.append(score)
                                break
                            else:
                                print("Lütfen 1-5 arası bir puan giriniz.")
                        except ValueError:
                            print("Geçerli bir sayı giriniz.")
            
            self.semantic_scores[model_name] = scores
            
            if scores:
                avg_score = np.mean(scores)
                print(f"Ortalama puan: {avg_score:.2f}")
    
    def automatic_semantic_evaluation(self):
        """Otomatik anlamsal değerlendirme (Demo için)"""
        print("\n" + "="*80)
        print("OTOMATİK ANLAMSAL DEĞERLENDİRME")
        print("="*80)
        
        self.semantic_scores = {}
        
        # Basit otomatik puanlama (benzerlik skoruna dayalı)
        for model_name, results in self.top5_results.items():
            scores = []
            
            if results:
                for title, similarity_score in results:
                    # Benzerlik skorunu 1-5 arasına dönüştürme
                    if similarity_score >= 0.8:
                        score = 5
                    elif similarity_score >= 0.6:
                        score = 4
                    elif similarity_score >= 0.4:
                        score = 3
                    elif similarity_score >= 0.2:
                        score = 2
                    else:
                        score = 1
                    scores.append(score)
            
            self.semantic_scores[model_name] = scores
            
            if scores:
                avg_score = np.mean(scores)
                print(f"{model_name}: Ortalama puan = {avg_score:.2f}")
    
    def calculate_jaccard_similarity(self, set1, set2):
        """Jaccard benzerliği hesaplama"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ranking_agreement(self):
        """Sıralama tutarlılığı değerlendirmesi"""
        print("\n" + "="*80)
        print("SIRALAMA TUTARLILIĞI DEĞERLENDİRMESİ")
        print("="*80)
        
        # Her model için film setlerini oluşturma
        model_sets = {}
        for model_name, results in self.top5_results.items():
            if results:
                model_sets[model_name] = set([title for title, _ in results])
            else:
                model_sets[model_name] = set()
        
        # Jaccard benzerlik matrisini hesaplama
        models = list(model_sets.keys())
        n_models = len(models)
        jaccard_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                jaccard_score = self.calculate_jaccard_similarity(model_sets[model1], model_sets[model2])
                jaccard_matrix[i][j] = jaccard_score
        
        self.jaccard_matrix = pd.DataFrame(jaccard_matrix, index=models, columns=models)
        
        print("Jaccard Benzerlik Matrisi:")
        print(self.jaccard_matrix.round(3))
        
        return self.jaccard_matrix
    
    def visualize_results(self):
        """Sonuçları görselleştirme"""
        # Anlamsal değerlendirme sonuçları
        if self.semantic_scores:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Ortalama puanlar
            model_names = list(self.semantic_scores.keys())
            avg_scores = [np.mean(scores) if scores else 0 for scores in self.semantic_scores.values()]
            
            if model_names and avg_scores:
                axes[0, 0].bar(range(len(model_names)), avg_scores)
                axes[0, 0].set_xticks(range(len(model_names)))
                axes[0, 0].set_xticklabels([name[:10] + "..." if len(name) > 10 else name for name in model_names], rotation=45, ha='right')
                axes[0, 0].set_title('Model Ortalama Anlamsal Puanları')
                axes[0, 0].set_ylabel('Ortalama Puan')
            
            # TF-IDF vs Word2Vec karşılaştırması
            tfidf_scores = []
            w2v_scores = []
            
            for model_name, scores in self.semantic_scores.items():
                if scores:
                    if 'tfidf' in model_name:
                        tfidf_scores.extend(scores)
                    else:
                        w2v_scores.extend(scores)
            
            if tfidf_scores and w2v_scores:
                axes[0, 1].boxplot([tfidf_scores, w2v_scores], labels=['TF-IDF', 'Word2Vec'])
                axes[0, 1].set_title('TF-IDF vs Word2Vec Karşılaştırması')
                axes[0, 1].set_ylabel('Anlamsal Puan')
            
            # Jaccard benzerlik matrisi
            if self.jaccard_matrix is not None and len(self.jaccard_matrix) > 0:
                # Matris boyutu çok büyükse ilk 10x10'u al
                matrix_to_plot = self.jaccard_matrix.iloc[:10, :10] if len(self.jaccard_matrix) > 10 else self.jaccard_matrix
                
                im = axes[1, 0].imshow(matrix_to_plot.values, cmap='Blues', aspect='auto')
                axes[1, 0].set_xticks(range(len(matrix_to_plot.columns)))
                axes[1, 0].set_yticks(range(len(matrix_to_plot.index)))
                
                # Kısa model adları
                short_cols = [name[:8] + "..." if len(name) > 8 else name for name in matrix_to_plot.columns]
                short_rows = [name[:8] + "..." if len(name) > 8 else name for name in matrix_to_plot.index]
                
                axes[1, 0].set_xticklabels(short_cols, rotation=45, ha='right')
                axes[1, 0].set_yticklabels(short_rows)
                axes[1, 0].set_title('Jaccard Benzerlik Matrisi')
                plt.colorbar(im, ax=axes[1, 0])
                
                # Matriste değerleri gösterme
                for i in range(len(matrix_to_plot.index)):
                    for j in range(len(matrix_to_plot.columns)):
                        axes[1, 0].text(j, i, f'{matrix_to_plot.iloc[i, j]:.2f}', 
                                       ha='center', va='center', fontsize=6)
            
            # Model türü analizi
            if len(avg_scores) > 2:
                axes[1, 1].hist(avg_scores, bins=min(5, len(avg_scores)), alpha=0.7)
                axes[1, 1].set_title('Anlamsal Puan Dağılımı')
                axes[1, 1].set_xlabel('Ortalama Puan')
                axes[1, 1].set_ylabel('Model Sayısı')
            
            plt.tight_layout()
            plt.savefig('similarity_evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Görselleştirme 'similarity_evaluation_results.png' dosyasına kaydedildi.")
    
    def generate_detailed_report(self, input_movie, output_file="similarity_evaluation_report.txt"):
        """Detaylı rapor oluşturma"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("METIN BENZERLİĞİ HESAPLAMA VE DEĞERLENDİRME RAPORU\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"GİRİŞ METNİ: {input_movie['title']}\n")
            f.write(f"ÖZET: {str(input_movie['original_text'])[:200]}...\n\n")
            
            f.write("1. MODEL SONUÇLARI VE BENZERLİK SKORLARI\n")
            f.write("-" * 50 + "\n\n")
            
            for model_name, results in self.top5_results.items():
                f.write(f"{model_name.upper()}:\n")
                if results:
                    for i, (title, score) in enumerate(results, 1):
                        f.write(f"  {i}. {title} (Skor: {score:.4f})\n")
                else:
                    f.write("  Sonuç bulunamadı\n")
                f.write("\n")
            
            f.write("2. ANLAMSAL DEĞERLENDİRME SONUÇLARI\n")
            f.write("-" * 50 + "\n\n")
            
            if self.semantic_scores:
                for model_name, scores in self.semantic_scores.items():
                    if scores:
                        avg_score = np.mean(scores)
                        f.write(f"{model_name}: {scores} → Ortalama: {avg_score:.2f}\n")
                    else:
                        f.write(f"{model_name}: Puan verilmedi\n")
            
            f.write("\n3. JACCARD BENZERLİK MATRİSİ\n")
            f.write("-" * 50 + "\n\n")
            
            if self.jaccard_matrix is not None:
                f.write(self.jaccard_matrix.round(3).to_string())
                f.write("\n\n")
            
            f.write("4. ANALİZ VE YORUMLAR\n")
            f.write("-" * 50 + "\n\n")
            
            # En iyi performans gösteren modeller
            if self.semantic_scores:
                best_models = [(name, np.mean(scores)) for name, scores in self.semantic_scores.items() if scores]
                best_models.sort(key=lambda x: x[1], reverse=True)
                
                f.write("En İyi Performans Gösteren Modeller:\n")
                for i, (model_name, avg_score) in enumerate(best_models[:3], 1):
                    f.write(f"{i}. {model_name}: {avg_score:.2f}\n")
                f.write("\n")
            
            # Jaccard analizi
            if self.jaccard_matrix is not None:
                # En yüksek Jaccard skorları (kendisi hariç)
                jaccard_values = []
                models = self.jaccard_matrix.index.tolist()
                
                for i, model1 in enumerate(models):
                    for j, model2 in enumerate(models):
                        if i != j:  # Kendisi hariç
                            jaccard_values.append((model1, model2, self.jaccard_matrix.iloc[i, j]))
                
                jaccard_values.sort(key=lambda x: x[2], reverse=True)
                
                f.write("En Yüksek Jaccard Benzerlik Skorları:\n")
                for i, (model1, model2, score) in enumerate(jaccard_values[:5], 1):
                    f.write(f"{i}. {model1} vs {model2}: {score:.3f}\n")
        
        print(f"Detaylı rapor '{output_file}' dosyasına kaydedildi.")
        return output_file

def main():
    """Ana fonksiyon"""
    print("METIN BENZERLİĞİ HESAPLAMA VE DEĞERLENDİRME SİSTEMİ")
    print("=" * 60)
    
    # Sistem oluşturma
    evaluator = SimilarityEvaluator()
    
    # Verileri ve modelleri yükleme
    if not evaluator.load_data_and_models():
        print("Sistem başlatılamadı. Veriler veya modeller eksik.")
        return
    
    print("\n" + "=" * 60)
    print("Mevcut filmler:")
    for i in range(min(10, len(evaluator.movie_data))):
        print(f"{i}: {evaluator.movie_data.iloc[i]['title']}")
    print("...")
    
    # Giriş metni seçme
    while True:
        choice = input("\nFilm seçimi yapınız:\n1. Film adı ile ara\n2. İndeks numarası ile seç\n3. Rastgele seç\nSeçiminiz (1/2/3): ")
        
        if choice == '1':
            movie_name = input("Film adını giriniz: ")
            input_movie = evaluator.select_input_text(movie_title=movie_name)
        elif choice == '2':
            try:
                movie_index = int(input("Film indeksini giriniz: "))
                input_movie = evaluator.select_input_text(movie_index=movie_index)
            except ValueError:
                print("Geçerli bir sayı giriniz.")
                continue
        elif choice == '3':
            input_movie = evaluator.select_input_text()
        else:
            print("Geçersiz seçim. Tekrar deneyiniz.")
            continue
        
        if input_movie is not None:
            break
    
    # Benzerlik hesaplamaları
    results = evaluator.calculate_all_similarities(input_movie)
    
    # Sonuçları görüntüleme
    evaluator.display_results()
    
    # Anlamsal değerlendirme
    eval_choice = input("\nAnlamsal değerlendirme yapmak ister misiniz?\n1. Manuel değerlendirme\n2. Otomatik değerlendirme\n3. Atla\nSeçiminiz (1/2/3): ")
    
    if eval_choice == '1':
        evaluator.semantic_evaluation()
    elif eval_choice == '2':
        evaluator.automatic_semantic_evaluation()
    
    # Sıralama tutarlılığı
    jaccard_matrix = evaluator.calculate_ranking_agreement()
    
    # Görselleştirme
    viz_choice = input("\nSonuçları görselleştirmek ister misiniz? (e/h): ")
    if viz_choice.lower() == 'e':
        evaluator.visualize_results()
    
    # Rapor oluşturma
    report_choice = input("\nDetaylı rapor oluşturmak ister misiniz? (e/h): ")
    if report_choice.lower() == 'e':
        report_file = evaluator.generate_detailed_report(input_movie)
        print(f"Rapor hazırlandı: {report_file}")
    
    print("\nAnaliz tamamlandı!")

if __name__ == "__main__":
    main()