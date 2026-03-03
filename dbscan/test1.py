import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
import time
import torch
import re
from datasets import load_dataset
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')

# =============================
# 🧩 SEED GLOBAL PER RIPRODUCIBILITÀ
# =============================
np.random.seed(42)
torch.manual_seed(42)

# =======================================================
# 📹 CARICAMENTO MODELLI STATICI (Word2Vec, FastText, GloVe)
# =======================================================
_STATIC_MODELS_CACHE = {}

def load_static_model(model_name):
    """Carica e cachea modelli statici."""
    if model_name not in _STATIC_MODELS_CACHE:
        print(f"    📥 Caricamento {model_name}...", end=" ", flush=True)
        if model_name == "word2vec":
            _STATIC_MODELS_CACHE[model_name] = api.load("word2vec-google-news-300")
        elif model_name == "fasttext":
            _STATIC_MODELS_CACHE[model_name] = api.load("fasttext-wiki-news-subwords-300")
        elif model_name == "glove":
            _STATIC_MODELS_CACHE[model_name] = api.load("glove-wiki-gigaword-100")
        print("✓")
    return _STATIC_MODELS_CACHE[model_name]

def get_static_embeddings(texts, model_name):
    """Genera embeddings per testi usando modelli statici."""
    model = load_static_model(model_name)
    embeddings = []
    missing_count = 0
    
    for text in texts:
        words = text.lower().split()[:100]
        word_embeddings = []
        
        for word in words:
            if word in model:
                word_embeddings.append(model[word])
            else:
                missing_count += 1
        
        if word_embeddings:
            embeddings.append(np.mean(word_embeddings, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    
    if missing_count > 0:
        print(f"      ⚠ {missing_count} parole non trovate nel vocabolario")
    
    return np.array(embeddings)

# =======================================================
# 📹 NORMALIZZAZIONE MIN-MAX
# =======================================================
def apply_minmax_normalization(embeddings):
    """Applica normalizzazione Min-Max feature-wise agli embeddings."""
    scaler = MinMaxScaler()
    embeddings_norm = scaler.fit_transform(embeddings)
    return embeddings_norm

# =======================================================
# 📹 CARICAMENTO DATASET
# =======================================================
def clean_text(text):
    """Pulisce il testo rimuovendo caratteri speciali e normalizzando."""
    text = re.sub(r'\s+', ' ', text)  # Rimuove spazi multipli
    text = text.strip()
    return text

def load_dataset_samples(dataset_name, n_samples=1000):
    """Carica un sottoinsieme di un dataset."""
    print(f"📦 Caricamento dataset: {dataset_name}...")
        
    if dataset_name.lower() == "ag_news":
        dataset = load_dataset('ag_news', split='train')
        dataset = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
        texts = dataset['text']
        labels = dataset['label']
        
    elif dataset_name.lower() == "20newsgroups":
        newsgroups = fetch_20newsgroups(
            subset='all', 
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        # Prendi un campione casuale
        indices = np.random.choice(len(newsgroups.data), min(n_samples, len(newsgroups.data)), replace=False)
        texts = [clean_text(newsgroups.data[i]) for i in indices]
        labels = [newsgroups.target[i] for i in indices]
        
        # Filtra testi troppo corti
        valid_indices = [i for i, text in enumerate(texts) if len(text.split()) >= 10]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        print(f"  Filtrati {len(valid_indices)} testi validi (>= 10 parole)")
        
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato")
    
    return texts, labels

# =======================================================
# 📹 GENERAZIONE EMBEDDINGS
# =======================================================
def generate_embeddings(model_name, texts, is_static=False):
    """Genera embeddings con modelli transformer o statici."""
    start_time = time.time()
    
    if is_static:
        embeddings = get_static_embeddings(texts, model_name)
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    
    enc_time = time.time() - start_time
    return embeddings, enc_time

# =======================================================
# 📹 CUSTOM K-MEANS CON COSINE E ADJUSTED COSINE
# =======================================================
class CosineKMeans:
    """K-Means con distanza cosine o adjusted cosine."""
    
    def __init__(self, n_clusters=8, metric='cosine', max_iter=300, n_init=10, random_state=42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        
    def _compute_distances(self, X, centers):
        """Calcola distanze cosine o adjusted cosine."""
        if self.metric == 'adjusted_cosine':
            # Centra i dati
            mean_features = np.mean(X, axis=0)
            X_centered = X - mean_features
            centers_centered = centers - mean_features
            return cosine_distances(X_centered, centers_centered)
        else:  # cosine
            return cosine_distances(X, centers)
    
    def _initialize_centers(self, X):
        """Inizializza centri usando k-means++."""
        n_samples = X.shape[0]
        np.random.seed(self.random_state)
        
        # Primo centro casuale
        centers = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Calcola distanze dai centri esistenti
            distances = self._compute_distances(X, np.array(centers))
            min_distances = np.min(distances, axis=1)
            
            # Probabilità proporzionali al quadrato delle distanze
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()
            
            # Scegli nuovo centro
            new_center_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(X[new_center_idx])
        
        return np.array(centers)
    
    def fit(self, X):
        """Esegue clustering K-Means con metrica cosine."""
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        
        for init_idx in range(self.n_init):
            # Inizializza centri
            centers = self._initialize_centers(X)
            
            for iteration in range(self.max_iter):
                # Assegna punti ai cluster
                distances = self._compute_distances(X, centers)
                labels = np.argmin(distances, axis=1)
                
                # Ricalcola centri
                new_centers = np.array([
                    X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
                    for k in range(self.n_clusters)
                ])
                
                # Check convergenza
                if np.allclose(centers, new_centers):
                    break
                    
                centers = new_centers
            
            # Calcola inertia (somma delle distanze minime)
            distances = self._compute_distances(X, centers)
            inertia = np.sum(np.min(distances, axis=1))
            
            # Salva migliore risultato
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
        
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        
        return self
    
    def fit_predict(self, X):
        """Esegue fit e ritorna le label."""
        self.fit(X)
        return self.labels_

# =======================================================
# 📹 VALUTAZIONE K-MEANS
# =======================================================
def evaluate_kmeans(embeddings, true_labels, n_clusters, metric_name):
    """
    Esegue K-Means e calcola le metriche di valutazione.
    """
    try:
        # Esegui K-Means
        if metric_name in ['cosine', 'adjusted_cosine']:
            kmeans = CosineKMeans(n_clusters=n_clusters, metric=metric_name, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        else:
            # Euclidean (fallback)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(embeddings)
        
        # Calcola metriche
        silhouette = silhouette_score(embeddings, labels, metric='euclidean')
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'adjusted_rand_index': ari,
            'nmi_score': nmi,
            'inertia': kmeans.inertia_ if hasattr(kmeans, 'inertia_') else None
        }
    except Exception as e:
        print(f"Errore: {e}")
        return None

def run_kmeans_experiments(embeddings, true_labels, embedding_model, dataset_name,
                           is_normalized, k_values):
    """Esegue esperimenti K-Means con diverse metriche e k."""
    norm_label = "NORMALIZED" if is_normalized else "RAW"
    print(f"    {'='*60}")
    print(f"    K-MEANS | Dataset: {dataset_name} | Embeddings: {norm_label}")
    print(f"    {'='*60}")
    
    results = []
    metrics = ['cosine', 'adjusted_cosine']
    
    for metric_name in metrics:
        print(f"      Metrica: {metric_name.upper()}")
        
        for k in k_values:
            print(f"        k={k}...", end=" ", flush=True)
            
            result = evaluate_kmeans(embeddings, true_labels, k, metric_name)
            
            if result is not None:
                result.update({
                    'dataset': dataset_name,
                    'embedding_model': embedding_model,
                    'normalized': is_normalized,
                    'metric': metric_name,
                    'algorithm': 'kmeans'
                })
                results.append(result)
                print(f"✓ sil={result['silhouette_score']:.3f}, ari={result['adjusted_rand_index']:.3f}")
            else:
                print("✗ clustering non valido")
    
    return results

# =======================================================
# 📹 VALUTAZIONE DBSCAN
# =======================================================
def evaluate_dbscan(embeddings, true_labels, eps, min_samples, metric_name):
    """Esegue DBSCAN e calcola le metriche di valutazione."""
    
    # Esegui DBSCAN
    if metric_name == 'adjusted_cosine':
        mean_features = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - mean_features
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
        labels = dbscan.fit_predict(centered_embeddings)
    else:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_name, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
    
    # Conta cluster (escludendo noise -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)
    
    if n_clusters < 2:
        return None
    
    # Filtra i punti noise
    mask = labels != -1
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    
    if len(filtered_labels) < 2 or len(set(filtered_labels)) < 2:
        return None
    
    try:
        silhouette = silhouette_score(filtered_embeddings, filtered_labels, metric='euclidean')
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
        calinski_harabasz = calinski_harabasz_score(filtered_embeddings, filtered_labels)
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'adjusted_rand_index': ari,
            'nmi_score': nmi,
        }
    except Exception as e:
        return None

def run_dbscan_experiments(embeddings, true_labels, embedding_model, dataset_name,
                           is_normalized, eps_values, min_samples):
    """Esegue esperimenti DBSCAN con diverse metriche e parametri."""
    norm_label = "NORMALIZED" if is_normalized else "RAW"
    print(f"    {'='*60}")
    print(f"    DBSCAN | Dataset: {dataset_name} | Embeddings: {norm_label}")
    print(f"    {'='*60}")
    
    results = []
    metrics = ['cosine', 'adjusted_cosine']
    
    for metric_name in metrics:
        print(f"      Metrica: {metric_name.upper()}")
        
        for eps in eps_values:
            print(f"        eps={eps:.2f}...", end=" ", flush=True)
            
            result = evaluate_dbscan(
                embeddings, true_labels, eps, min_samples, metric_name
            )
            
            if result is not None:
                result.update({
                    'dataset': dataset_name,
                    'embedding_model': embedding_model,
                    'normalized': is_normalized,
                    'metric': metric_name,
                    'eps': eps,
                    'min_samples': min_samples,
                    'algorithm': 'dbscan'
                })
                results.append(result)
                print(f"✓ clusters={result['n_clusters']}, noise={result['n_noise']}, "
                      f"sil={result['silhouette_score']:.3f}, ari={result['adjusted_rand_index']:.3f}")
            else:
                print("✗ clustering non valido")
    
    return results

# =======================================================
# 📹 PIPELINE PRINCIPALE
# =======================================================
def main():
    print("="*90)
    print("DBSCAN + K-MEANS: DATASETS × EMBEDDING MODELS × 2 METRICHE × NORMALIZZAZIONE")
    print("="*90)

    N_SAMPLES = 1000
    MIN_SAMPLES = 5
    EPS_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Dataset
    DATASETS = ["ag_news", "20newsgroups"]
    
    EMBEDDING_MODELS = [
        # Modelli transformer
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L6-v2",
        "paraphrase-albert-small-v2",
        # Modelli statici
        "word2vec",
        "fasttext",
        "glove"
    ]
    
    STATIC_MODELS = ["word2vec", "fasttext", "glove"]

    all_results = []

    for dataset_name in DATASETS:
        print(f"\n{'='*90}\nDATASET: {dataset_name.upper()}\n{'='*90}")
        
        try:
            texts, labels = load_dataset_samples(dataset_name, n_samples=N_SAMPLES)
            
            # Info dataset
            unique_labels = len(set(labels))
            print(f"  Samples: {len(texts)} | Classi: {unique_labels}")
            label_dist = pd.Series(labels).value_counts().to_dict()
            print(f"  Distribuzione: {label_dist}")
            
            # K-values per K-Means basati sul numero di classi reali
            K_VALUES = [max(2, unique_labels - 2), unique_labels, unique_labels + 2, unique_labels + 5]
            K_VALUES = sorted(set(K_VALUES))  # Rimuovi duplicati
            
            for model_name in EMBEDDING_MODELS:
                print(f"\n  {'─'*80}\n  EMBEDDING: {model_name}\n  {'─'*80}")
                
                try:
                    is_static = model_name in STATIC_MODELS
                    embeddings, enc_time = generate_embeddings(model_name, texts, is_static)
                    
                    print(f"    Embedding shape: {embeddings.shape}, encoding time: {enc_time:.2f}s")
                    
                    # ===== DBSCAN RAW =====
                    results_dbscan_raw = run_dbscan_experiments(
                        embeddings, labels, model_name, dataset_name,
                        is_normalized=False, eps_values=EPS_VALUES, 
                        min_samples=MIN_SAMPLES
                    )
                    all_results.extend(results_dbscan_raw)
                    
                    # ===== DBSCAN NORMALIZED =====
                    embeddings_norm = apply_minmax_normalization(embeddings)
                    results_dbscan_norm = run_dbscan_experiments(
                        embeddings_norm, labels, model_name, dataset_name,
                        is_normalized=True, eps_values=EPS_VALUES,
                        min_samples=MIN_SAMPLES
                    )
                    all_results.extend(results_dbscan_norm)
                    
                    # ===== K-MEANS RAW =====
                    results_kmeans_raw = run_kmeans_experiments(
                        embeddings, labels, model_name, dataset_name,
                        is_normalized=False, k_values=K_VALUES
                    )
                    all_results.extend(results_kmeans_raw)
                    
                    # ===== K-MEANS NORMALIZED =====
                    results_kmeans_norm = run_kmeans_experiments(
                        embeddings_norm, labels, model_name, dataset_name,
                        is_normalized=True, k_values=K_VALUES
                    )
                    all_results.extend(results_kmeans_norm)
                    
                except Exception as e:
                    print(f"  ✗ Errore con modello {model_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"✗ Errore con dataset {dataset_name}: {e}")
            continue

    # =======================================================
    # 📹 ANALISI RISULTATI
    # =======================================================
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n\n" + "="*90)
        print("📊 RISULTATI COMPLETI")
        print("="*90)
        
        # Statistiche per algoritmo
        print("\n" + "="*90)
        print("📈 STATISTICHE PER ALGORITMO")
        print("="*90)
        
        for algo in df['algorithm'].unique():
            df_algo = df[df['algorithm'] == algo]
            print(f"\n{algo.upper()}:")
            print(f"  Configurazioni valide: {len(df_algo)}")
            print(f"  Avg Silhouette:  {df_algo['silhouette_score'].mean():.4f} (±{df_algo['silhouette_score'].std():.4f})")
            print(f"  Avg ARI:         {df_algo['adjusted_rand_index'].mean():.4f} (±{df_algo['adjusted_rand_index'].std():.4f})")
            print(f"  Avg NMI:         {df_algo['nmi_score'].mean():.4f} (±{df_algo['nmi_score'].std():.4f})")
        
        # Confronto DBSCAN vs K-Means
        print("\n\n" + "="*90)
        print("🔄 CONFRONTO DBSCAN vs K-MEANS")
        print("="*90)
        
        for metric in ['cosine', 'adjusted_cosine']:
            print(f"\n{metric.upper()}:")
            df_metric = df[df['metric'] == metric]
            
            for algo in ['dbscan', 'kmeans']:
                df_algo = df_metric[df_metric['algorithm'] == algo]
                if len(df_algo) > 0:
                    print(f"  {algo.upper()}:")
                    print(f"    Silhouette: {df_algo['silhouette_score'].mean():.4f}")
                    print(f"    ARI:        {df_algo['adjusted_rand_index'].mean():.4f}")
                    print(f"    NMI:        {df_algo['nmi_score'].mean():.4f}")
        
        # Best configurations per algoritmo
        print("\n\n" + "="*90)
        print("🏆 MIGLIORI CONFIGURAZIONI PER ALGORITMO")
        print("="*90)
        
        for algo in df['algorithm'].unique():
            df_algo = df[df['algorithm'] == algo]
            print(f"\n{algo.upper()} - Migliore ARI:")
            
            idx = df_algo['adjusted_rand_index'].idxmax()
            best = df_algo.loc[idx]
            norm_status = "NORMALIZED" if best['normalized'] else "RAW"
            
            print(f"  ARI: {best['adjusted_rand_index']:.4f}")
            print(f"  Dataset: {best['dataset']}")
            print(f"  Model: {best['embedding_model']}")
            print(f"  Metric: {best['metric']}")
            print(f"  Embeddings: {norm_status}")
            print(f"  Silhouette: {best['silhouette_score']:.4f}")
            print(f"  NMI: {best['nmi_score']:.4f}")
        
        # Salva risultati
        output_file = 'clustering_multi_algo_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Risultati completi salvati in '{output_file}'")
        
        # Summary
        summary_df = df.groupby(['algorithm', 'dataset', 'embedding_model', 'normalized', 'metric']).agg({
            'silhouette_score': ['mean', 'std', 'max'],
            'adjusted_rand_index': ['mean', 'std', 'max'],
            'nmi_score': ['mean', 'std', 'max']
        }).round(4)
        
        summary_file = 'clustering_summary_results.csv'
        summary_df.to_csv(summary_file)
        print(f"✅ Summary salvato in '{summary_file}'")
        
        return df
    else:
        print("\n✗ Nessun risultato generato.")
        return None

if __name__ == "__main__":
    main()