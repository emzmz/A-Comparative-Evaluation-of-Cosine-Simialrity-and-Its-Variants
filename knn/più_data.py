import numpy as np
import pandas as pd
import time
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, normalize, label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import warnings
warnings.filterwarnings("ignore")

# =============================
# 🧩 SEED GLOBAL PER RIPRODUCIBILITÀ
# =============================
np.random.seed(42)
torch.manual_seed(42)

# =======================================================
# 📹 CARICAMENTO MODELLI STATICI
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
# 📹 CUSTOM CLASSIFIERS CON COSINE E ADJUSTED COSINE
# =======================================================

class AdjustedCosineKNN:
    """KNN basato su Adjusted Cosine Similarity."""
    
    def __init__(self, n_neighbors=5, batch_size=256):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.mean_features = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        X_centered = X - self.mean_features
        X_train_centered = self.X_train - self.mean_features

        norm_train = np.linalg.norm(X_train_centered, axis=1, keepdims=True)
        norm_train[norm_train == 0] = 1.0
        X_train_normed = X_train_centered / norm_train

        predictions = []
        for start in range(0, len(X_centered), self.batch_size):
            end = start + self.batch_size
            batch = X_centered[start:end]
            norm_batch = np.linalg.norm(batch, axis=1, keepdims=True)
            norm_batch[norm_batch == 0] = 1.0
            batch_normed = batch / norm_batch

            sims = np.dot(batch_normed, X_train_normed.T)
            k_idx = np.argpartition(-sims, self.n_neighbors-1, axis=1)[:, :self.n_neighbors]
            
            for i, neighbors_idx in enumerate(k_idx):
                neighbor_labels = self.y_train[neighbors_idx]
                pred = np.bincount(neighbor_labels).argmax()
                predictions.append(pred)

        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities (approximated via voting)."""
        X = np.array(X, dtype=np.float32)
        X_centered = X - self.mean_features
        X_train_centered = self.X_train - self.mean_features

        norm_train = np.linalg.norm(X_train_centered, axis=1, keepdims=True)
        norm_train[norm_train == 0] = 1.0
        X_train_normed = X_train_centered / norm_train

        n_classes = len(np.unique(self.y_train))
        probabilities = []
        
        for start in range(0, len(X_centered), self.batch_size):
            end = start + self.batch_size
            batch = X_centered[start:end]
            norm_batch = np.linalg.norm(batch, axis=1, keepdims=True)
            norm_batch[norm_batch == 0] = 1.0
            batch_normed = batch / norm_batch

            sims = np.dot(batch_normed, X_train_normed.T)
            k_idx = np.argpartition(-sims, self.n_neighbors-1, axis=1)[:, :self.n_neighbors]
            
            for i, neighbors_idx in enumerate(k_idx):
                neighbor_labels = self.y_train[neighbors_idx]
                counts = np.bincount(neighbor_labels, minlength=n_classes)
                proba = counts / self.n_neighbors
                probabilities.append(proba)

        return np.array(probabilities)


class CosineLogisticRegression:
    """Logistic Regression che usa cosine similarity come features."""
    
    def __init__(self, metric='cosine', C=1.0, max_iter=1000):
        self.metric = metric
        self.C = C
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        self.lr = None
    
    def _compute_similarities(self, X, X_ref):
        """Calcola similarities tra X e X_ref."""
        if self.metric == 'adjusted_cosine':
            X_centered = X - self.mean_features
            X_ref_centered = X_ref - self.mean_features
            return cosine_similarity(X_centered, X_ref_centered)
        else:  # cosine
            return cosine_similarity(X, X_ref)
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        
        # Calcola similarity matrix come features
        sim_features = self._compute_similarities(self.X_train, self.X_train)
        
        # Allena Logistic Regression sulle similarities
        self.lr = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=42)
        self.lr.fit(sim_features, self.y_train)
        return self
    
    def predict(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.lr.predict(sim_features)
    
    def predict_proba(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.lr.predict_proba(sim_features)


class CosineSVM:
    """Linear SVM che usa cosine similarity come features."""
    
    def __init__(self, metric='cosine', C=1.0, max_iter=1000):
        self.metric = metric
        self.C = C
        self.max_iter = max_iter
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        self.svm = None
    
    def _compute_similarities(self, X, X_ref):
        if self.metric == 'adjusted_cosine':
            X_centered = X - self.mean_features
            X_ref_centered = X_ref - self.mean_features
            return cosine_similarity(X_centered, X_ref_centered)
        else:
            return cosine_similarity(X, X_ref)
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        
        sim_features = self._compute_similarities(self.X_train, self.X_train)
        
        self.svm = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=42, dual='auto')
        self.svm.fit(sim_features, self.y_train)
        return self
    
    def predict(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.svm.predict(sim_features)
    
    def decision_function(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.svm.decision_function(sim_features)


class CosineNaiveBayes:
    """Naive Bayes che usa cosine similarity come features."""
    
    def __init__(self, metric='cosine'):
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        self.nb = None
    
    def _compute_similarities(self, X, X_ref):
        if self.metric == 'adjusted_cosine':
            X_centered = X - self.mean_features
            X_ref_centered = X_ref - self.mean_features
            # Shift to positive values for Naive Bayes
            sims = cosine_similarity(X_centered, X_ref_centered)
            return (sims + 1) / 2  # Scale from [-1,1] to [0,1]
        else:
            sims = cosine_similarity(X, X_ref)
            return (sims + 1) / 2
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        
        sim_features = self._compute_similarities(self.X_train, self.X_train)
        
        self.nb = GaussianNB()
        self.nb.fit(sim_features, self.y_train)
        return self
    
    def predict(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.nb.predict(sim_features)
    
    def predict_proba(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.nb.predict_proba(sim_features)


class CosineDecisionTree:
    """Decision Tree che usa cosine similarity come features."""
    
    def __init__(self, metric='cosine', max_depth=None):
        self.metric = metric
        self.max_depth = max_depth
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        self.dt = None
    
    def _compute_similarities(self, X, X_ref):
        if self.metric == 'adjusted_cosine':
            X_centered = X - self.mean_features
            X_ref_centered = X_ref - self.mean_features
            return cosine_similarity(X_centered, X_ref_centered)
        else:
            return cosine_similarity(X, X_ref)
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        
        sim_features = self._compute_similarities(self.X_train, self.X_train)
        
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        self.dt.fit(sim_features, self.y_train)
        return self
    
    def predict(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.dt.predict(sim_features)
    
    def predict_proba(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.dt.predict_proba(sim_features)


class CosineRandomForest:
    """Random Forest che usa cosine similarity come features."""
    
    def __init__(self, metric='cosine', n_estimators=100, max_depth=None):
        self.metric = metric
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        self.rf = None
    
    def _compute_similarities(self, X, X_ref):
        if self.metric == 'adjusted_cosine':
            X_centered = X - self.mean_features
            X_ref_centered = X_ref - self.mean_features
            return cosine_similarity(X_centered, X_ref_centered)
        else:
            return cosine_similarity(X, X_ref)
    
    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        
        sim_features = self._compute_similarities(self.X_train, self.X_train)
        
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth, 
            random_state=42,
            n_jobs=-1
        )
        self.rf.fit(sim_features, self.y_train)
        return self
    
    def predict(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.rf.predict(sim_features)
    
    def predict_proba(self, X):
        sim_features = self._compute_similarities(X, self.X_train)
        return self.rf.predict_proba(sim_features)


# =======================================================
# 📹 NORMALIZZAZIONE MIN-MAX
# =======================================================
def apply_minmax_normalization(X_train, X_test):
    """Applica normalizzazione Min-Max feature-wise agli embeddings."""
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

# =======================================================
# 📹 CARICAMENTO DATASET
# =======================================================
def load_dataset_samples(dataset_name, n_samples=500, test_size=0.3):
    """Carica un sottoinsieme di un dataset."""
    print(f"📦 Caricamento dataset: {dataset_name}...")
    
    if dataset_name.lower() == "imdb":
        dataset = load_dataset('imdb', split='train')
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        texts = dataset['text']
        labels = dataset['label']
        
    elif dataset_name.lower() == "ag_news":
        dataset = load_dataset('ag_news', split='train')
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        texts = dataset['text']
        labels = dataset['label']
        
    elif dataset_name.lower() == "sst2":
        dataset = load_dataset('glue', 'sst2', split='train')
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        texts = dataset['sentence']
        labels = dataset['label']
        
    elif dataset_name.lower() == "dbpedia_14":
        dataset = load_dataset('dbpedia_14', split='train')
        n_samples = min(n_samples, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        texts = dataset['content']
        labels = dataset['label']
        
    elif dataset_name.lower() == "yelp_polarity":
        dataset = load_dataset('yelp_polarity', split='train')
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
        texts = dataset['text']
        labels = dataset['label']
        
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato")
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# =======================================================
# 📹 FUNZIONI AUSILIARIE
# =======================================================
def generate_embeddings(model_name, texts, is_static=False):
    """Genera embeddings con modelli transformer o statici."""
    start = time.time()
    
    if is_static:
        embeddings = get_static_embeddings(texts, model_name)
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    
    elapsed = time.time() - start
    return embeddings, elapsed

def compute_auc(model, X_test, y_test):
    """Calcola AUC in modo appropriato per classificazione binaria o multi-classe."""
    n_classes = len(np.unique(y_test))
    
    try:
        if n_classes == 2:
            # Classificazione binaria
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                return None
            
            auc = roc_auc_score(y_test, y_proba)
            
        else:
            # Classificazione multi-classe (OVR)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                return None
            
            # Binarizza le labels
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            
            # Calcola AUC macro-averaged
            auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        
        return auc
    except Exception as e:
        return None

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Allena, valuta e misura tempi e metriche di un modello."""
    print(f"      ▶ {model_name}...", end=" ", flush=True)
    
    try:
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_pred

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calcola AUC
        auc = compute_auc(model, X_test, y_test)
        
        auc_str = f", AUC: {auc:.4f}" if auc is not None else ""
        print(f"✓ Acc: {acc:.4f}, F1: {f1:.4f}{auc_str}")

        return {
            'accuracy': acc, 
            'precision': prec, 
            'recall': rec, 
            'f1': f1,
            'auc': auc if auc is not None else np.nan,
            'train_time': train_time, 
            'pred_time': pred_time
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def run_experiments(X_train, X_test, y_train, y_test, embedding_model, dataset_name, 
                   normalized=False, k_neighbors=5):
    """Esegue tutti i classificatori con 2 metriche (Cosine e Adjusted Cosine)."""
    norm_label = "NORMALIZED" if normalized else "RAW"
    print(f"    {'='*60}")
    print(f"    Dataset: {dataset_name} | Embeddings: {norm_label}")
    print(f"    {'='*60}")

    results = []
    
    # Definizione classificatori
    classifiers = {
        'KNN': [
            ('cosine', KNeighborsClassifier(n_neighbors=k_neighbors, metric='cosine', algorithm='brute', n_jobs=-1)),
            ('adjusted_cosine', AdjustedCosineKNN(n_neighbors=k_neighbors))
        ],
        'LogisticRegression': [
            ('cosine', CosineLogisticRegression(metric='cosine', C=1.0)),
            ('adjusted_cosine', CosineLogisticRegression(metric='adjusted_cosine', C=1.0))
        ],
        'SVM': [
            ('cosine', CosineSVM(metric='cosine', C=1.0)),
            ('adjusted_cosine', CosineSVM(metric='adjusted_cosine', C=1.0))
        ],
        'NaiveBayes': [
            ('cosine', CosineNaiveBayes(metric='cosine')),
            ('adjusted_cosine', CosineNaiveBayes(metric='adjusted_cosine'))
        ],
        'DecisionTree': [
            ('cosine', CosineDecisionTree(metric='cosine', max_depth=10)),
            ('adjusted_cosine', CosineDecisionTree(metric='adjusted_cosine', max_depth=10))
        ],
        'RandomForest': [
            ('cosine', CosineRandomForest(metric='cosine', n_estimators=50, max_depth=10)),
            ('adjusted_cosine', CosineRandomForest(metric='adjusted_cosine', n_estimators=50, max_depth=10))
        ]
    }
    
    for clf_name, clf_variants in classifiers.items():
        print(f"\n    {clf_name}:")
        
        for metric_name, model in clf_variants:
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, metric_name)
            
            if metrics is not None:
                result = {
                    'dataset': dataset_name,
                    'embedding_model': embedding_model,
                    'normalized': normalized,
                    'classifier': clf_name,
                    'metric': metric_name,
                    'embedding_dim': X_train.shape[1]
                }
                result.update(metrics)
                results.append(result)
    
    return results

# =======================================================
# 📹 PIPELINE PRINCIPALE
# =======================================================
def main():
    print("="*90)
    print("MULTI-CLASSIFIER COMPARISON: DATASETS × EMBEDDINGS × CLASSIFIERS × METRICS × NORMALIZATION")
    print("="*90)

    N_SAMPLES = 500
    TEST_SIZE = 0.3
    K_NEIGHBORS = 5
    
    DATASETS = ["imdb", "ag_news", "sst2", "dbpedia_14", "yelp_polarity"]
    
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
            X_train_text, X_test_text, y_train, y_test = load_dataset_samples(
                dataset_name, n_samples=N_SAMPLES, test_size=TEST_SIZE
            )
            print(f"  Train: {len(X_train_text)} | Test: {len(X_test_text)}")
            
            for model_name in EMBEDDING_MODELS:
                print(f"\n  {'─'*80}\n  EMBEDDING: {model_name}\n  {'─'*80}")
                
                try:
                    is_static = model_name in STATIC_MODELS
                    X_train_emb, _ = generate_embeddings(model_name, X_train_text, is_static)
                    X_test_emb, _ = generate_embeddings(model_name, X_test_text, is_static)
                    
                    # ===== VALUTAZIONE EMBEDDINGS RAW =====
                    results_raw = run_experiments(
                        X_train_emb, X_test_emb, y_train, y_test,
                        embedding_model=model_name, dataset_name=dataset_name,
                        normalized=False, k_neighbors=K_NEIGHBORS
                    )
                    all_results.extend(results_raw)
                    
                    # ===== VALUTAZIONE EMBEDDINGS NORMALIZZATI =====
                    X_train_norm, X_test_norm = apply_minmax_normalization(X_train_emb, X_test_emb)
                    results_norm = run_experiments(
                        X_train_norm, X_test_norm, y_train, y_test,
                        embedding_model=model_name, dataset_name=dataset_name,
                        normalized=True, k_neighbors=K_NEIGHBORS
                    )
                    all_results.extend(results_norm)
                    
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
        
        summary_cols = ['dataset', 'embedding_model', 'classifier', 'metric', 
                       'normalized', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        
        print("\nPrime 30 configurazioni:")
        print(df[summary_cols].head(30).to_string(index=False))
        
        # Statistiche per classificatore
        print("\n\n" + "="*90)
        print("📈 STATISTICHE PER CLASSIFICATORE")
        print("="*90)
        
        for clf in df['classifier'].unique():
            df_clf = df[df['classifier'] == clf]
            print(f"\n{clf}:")
            print(f"  Configurazioni: {len(df_clf)}")
            print(f"  Avg Accuracy:   {df_clf['accuracy'].mean():.4f} (±{df_clf['accuracy'].std():.4f})")
            print(f"  Avg Precision:  {df_clf['precision'].mean():.4f} (±{df_clf['precision'].std():.4f})")
            print(f"  Avg Recall:     {df_clf['recall'].mean():.4f} (±{df_clf['recall'].std():.4f})")
            print(f"  Avg F1:         {df_clf['f1'].mean():.4f} (±{df_clf['f1'].std():.4f})")
            if df_clf['auc'].notna().any():
                print(f"  Avg AUC:        {df_clf['auc'].mean():.4f} (±{df_clf['auc'].std():.4f})")
            print(f"  Best Accuracy:  {df_clf['accuracy'].max():.4f}")
        
        # Statistiche per modello di embedding
        print("\n\n" + "="*90)
        print("🔤 STATISTICHE PER MODELLO DI EMBEDDING")
        print("="*90)
        
        for emb in df['embedding_model'].unique():
            df_emb = df[df['embedding_model'] == emb]
            print(f"\n{emb}:")
            print(f"  Configurazioni: {len(df_emb)}")
            print(f"  Avg Accuracy:   {df_emb['accuracy'].mean():.4f} (±{df_emb['accuracy'].std():.4f})")
            print(f"  Avg Precision:  {df_emb['precision'].mean():.4f} (±{df_emb['precision'].std():.4f})")
            print(f"  Avg Recall:     {df_emb['recall'].mean():.4f} (±{df_emb['recall'].std():.4f})")
            print(f"  Avg F1:         {df_emb['f1'].mean():.4f} (±{df_emb['f1'].std():.4f})")
            if df_emb['auc'].notna().any():
                print(f"  Avg AUC:        {df_emb['auc'].mean():.4f} (±{df_emb['auc'].std():.4f})")
            print(f"  Best Accuracy:  {df_emb['accuracy'].max():.4f}")
        
        # Confronto Cosine vs Adjusted Cosine
        print("\n\n" + "="*90)
        print("🔄 CONFRONTO COSINE vs ADJUSTED COSINE")
        print("="*90)
        
        for clf in df['classifier'].unique():
            df_clf = df[df['classifier'] == clf]
            
            cosine_acc = df_clf[df_clf['metric'] == 'cosine']['accuracy'].mean()
            adjusted_acc = df_clf[df_clf['metric'] == 'adjusted_cosine']['accuracy'].mean()
            
            if not np.isnan(cosine_acc) and not np.isnan(adjusted_acc):
                improvement = ((adjusted_acc - cosine_acc) / cosine_acc) * 100
                print(f"\n{clf}:")
                print(f"  Cosine:          {cosine_acc:.4f}")
                print(f"  Adjusted Cosine: {adjusted_acc:.4f}")
                print(f"  Variazione:      {improvement:+.2f}%")
        
        # Confronto Raw vs Normalized
        print("\n\n" + "="*90)
        print("📊 ANALISI IMPATTO NORMALIZZAZIONE")
        print("="*90)
        
        for clf in df['classifier'].unique():
            df_clf = df[df['classifier'] == clf]
            
            raw_acc = df_clf[df_clf['normalized'] == False]['accuracy'].mean()
            norm_acc = df_clf[df_clf['normalized'] == True]['accuracy'].mean()
            
            if not np.isnan(raw_acc) and not np.isnan(norm_acc):
                improvement = ((norm_acc - raw_acc) / raw_acc) * 100
                print(f"\n{clf}:")
                print(f"  RAW:        {raw_acc:.4f}")
                print(f"  NORMALIZED: {norm_acc:.4f}")
                print(f"  Variazione: {improvement:+.2f}%")
        
        # Best configurations
        print("\n\n" + "="*90)
        print("🏆 MIGLIORI CONFIGURAZIONI")
        print("="*90)
        
        # Overall best
        idx = df['accuracy'].idxmax()
        best = df.loc[idx]
        norm_status = "NORMALIZED" if best['normalized'] else "RAW"
        
        print(f"\nMIGLIORE ASSOLUTA (Accuracy: {best['accuracy']:.4f}):")
        print(f"  Dataset:    {best['dataset']}")
        print(f"  Embedding:  {best['embedding_model']}")
        print(f"  Classifier: {best['classifier']}")
        print(f"  Metric:     {best['metric']}")
        print(f"  Normalized: {norm_status}")
        print(f"  Precision:  {best['precision']:.4f}")
        print(f"  Recall:     {best['recall']:.4f}")
        print(f"  F1 Score:   {best['f1']:.4f}")
        if not np.isnan(best['auc']):
            print(f"  AUC:        {best['auc']:.4f}")
        
        # Best per classifier
        print("\n\nMIGLIORI PER CLASSIFICATORE:")
        for clf in df['classifier'].unique():
            df_clf = df[df['classifier'] == clf]
            idx = df_clf['accuracy'].idxmax()
            best = df_clf.loc[idx]
            norm_status = "NORMALIZED" if best['normalized'] else "RAW"
            
            print(f"\n{clf}:")
            print(f"  Accuracy:   {best['accuracy']:.4f}")
            print(f"  Precision:  {best['precision']:.4f}")
            print(f"  Recall:     {best['recall']:.4f}")
            print(f"  F1:         {best['f1']:.4f}")
            print(f"  Dataset:    {best['dataset']}")
            print(f"  Embedding:  {best['embedding_model']}")
            print(f"  Metric:     {best['metric']}")
            print(f"  Normalized: {norm_status}")
        
        # Best per embedding model
        print("\n\nMIGLIORI PER MODELLO DI EMBEDDING:")
        for emb in df['embedding_model'].unique():
            df_emb = df[df['embedding_model'] == emb]
            idx = df_emb['accuracy'].idxmax()
            best = df_emb.loc[idx]
            norm_status = "NORMALIZED" if best['normalized'] else "RAW"
            
            print(f"\n{emb}:")
            print(f"  Accuracy:   {best['accuracy']:.4f}")
            print(f"  Precision:  {best['precision']:.4f}")
            print(f"  Recall:     {best['recall']:.4f}")
            print(f"  F1:         {best['f1']:.4f}")
            print(f"  Dataset:    {best['dataset']}")
            print(f"  Classifier: {best['classifier']}")
            print(f"  Metric:     {best['metric']}")
            print(f"  Normalized: {norm_status}")
        
        # Salva risultati
        output_file = 'multi_classifier_comparison_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Risultati completi salvati in '{output_file}'")
        
        # Summary aggregato per classificatore
        summary_clf = df.groupby(['classifier', 'metric', 'normalized']).agg({
            'accuracy': ['mean', 'std', 'max'],
            'precision': ['mean', 'std', 'max'],
            'recall': ['mean', 'std', 'max'],
            'f1': ['mean', 'std', 'max'],
            'auc': ['mean', 'std', 'max'],
            'train_time': 'mean',
            'pred_time': 'mean'
        }).round(4)
        
        summary_clf_file = 'classifier_summary_results.csv'
        summary_clf.to_csv(summary_clf_file)
        print(f"✅ Summary per classificatore salvato in '{summary_clf_file}'")
        
        # Summary aggregato per embedding model
        summary_emb = df.groupby(['embedding_model', 'metric', 'normalized']).agg({
            'accuracy': ['mean', 'std', 'max'],
            'precision': ['mean', 'std', 'max'],
            'recall': ['mean', 'std', 'max'],
            'f1': ['mean', 'std', 'max'],
            'auc': ['mean', 'std', 'max'],
            'train_time': 'mean',
            'pred_time': 'mean'
        }).round(4)
        
        summary_emb_file = 'embedding_summary_results.csv'
        summary_emb.to_csv(summary_emb_file)
        print(f"✅ Summary per embedding model salvato in '{summary_emb_file}'")
        
        return df
    else:
        print("\n✗ Nessun risultato generato.")
        return None

if __name__ == "__main__":
    main()