import numpy as np
import pandas as pd
import time
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# =============================
# 🧩 SEED GLOBAL PER RIPRODUCIBILITÀ
# =============================
np.random.seed(42)
torch.manual_seed(42)

# =======================================================
# 🔹 AdjustedCosineKNN (VERSIONE VETTORIALIZZATA & VELOCE)
# =======================================================
class AdjustedCosineKNN:
    """KNN basato su Adjusted Cosine Similarity (centrando i vettori sulle feature mean del training)."""
    
    def __init__(self, n_neighbors=5, batch_size=256):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.mean_features = None

    def fit(self, X, y):
        """Addestra il modello (memorizza il training set e la media feature-wise)."""
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.mean_features = np.mean(self.X_train, axis=0, keepdims=True)
        return self

    def predict(self, X):
        """Predice le etichette per il test set in modo vettoriale (batching per memoria)."""
        X = np.array(X, dtype=np.float32)
        X_centered = X - self.mean_features
        X_train_centered = self.X_train - self.mean_features

        # Normalizza i vettori
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

            # Similarità del coseno centrata
            sims = np.dot(batch_normed, X_train_normed.T)  # (batch, n_train)

            # Trova i top-k per riga
            k_idx = np.argpartition(-sims, self.n_neighbors-1, axis=1)[:, :self.n_neighbors]
            for i, neighbors_idx in enumerate(k_idx):
                neighbor_labels = self.y_train[neighbors_idx]
                pred = np.bincount(neighbor_labels).argmax()
                predictions.append(pred)

        return np.array(predictions)


# =======================================================
# 🔹 FUNZIONI AUSILIARIE
# =======================================================
def load_and_prepare_imdb(n_samples=2000):
    """Carica e prepara un sottoinsieme del dataset IMDB."""
    print("📦 Caricamento dataset IMDB...")
    dataset = load_dataset('imdb', split='train')
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    texts = dataset['text']
    labels = dataset['label']
    return texts, labels


def generate_embeddings(model, texts):
    """Genera embeddings con un modello già caricato."""
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.time() - start
    return embeddings, elapsed


def normalize_embeddings_by_train(train_embeddings, test_embeddings):
    """Normalizza embeddings (Z-score) usando SOLO le statistiche del training."""
    train_mean = np.mean(train_embeddings, axis=0)
    train_std = np.std(train_embeddings, axis=0)
    train_std[train_std == 0] = 1.0
    train_norm = (train_embeddings - train_mean) / train_std
    test_norm = (test_embeddings - train_mean) / train_std
    return train_norm, test_norm


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Allena, valuta e misura tempi e metriche di un modello."""
    print(f"    ▶ {model_name}...", end=" ", flush=True)
    
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"✓ Acc: {acc:.4f}, F1: {f1:.4f}")

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'train_time': train_time, 'pred_time': pred_time
    }


def run_experiments_on_embeddings(X_train, X_test, y_train, y_test, 
                                   embedding_model, is_normalized, encoding_time, k_neighbors=5):
    """Esegue KNN con Cosine e Adjusted Cosine e ritorna le metriche."""
    norm_label = "Normalizzati" if is_normalized else "Standard"
    print(f"\n  {'='*70}\n  Embeddings {norm_label}\n  {'='*70}")

    results = {
        'embedding_model': embedding_model,
        'normalized': is_normalized,
        'encoding_time': encoding_time,
        'embedding_dim': X_train.shape[1]
    }

    # KNN Cosine (scikit-learn)
    knn_cosine = KNeighborsClassifier(
        n_neighbors=k_neighbors,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1,
        weights='distance'
    )
    cosine_metrics = evaluate_model(knn_cosine, X_train, X_test, y_train, y_test, "KNN Cosine")
    for k, v in cosine_metrics.items():
        results[f'cosine_{k}'] = v

    # Adjusted Cosine KNN
    knn_adjusted = AdjustedCosineKNN(n_neighbors=k_neighbors)
    adjusted_metrics = evaluate_model(knn_adjusted, X_train, X_test, y_train, y_test, "KNN Adjusted Cosine")
    for k, v in adjusted_metrics.items():
        results[f'adjusted_{k}'] = v

    return results


def create_comparison_table(all_results):
    """Crea una tabella comparativa dei risultati."""
    df = pd.DataFrame(all_results)
    cols_order = [
        'embedding_model', 'normalized', 'embedding_dim', 'encoding_time',
        'cosine_accuracy', 'adjusted_accuracy',
        'cosine_f1', 'adjusted_f1',
        'cosine_precision', 'adjusted_precision',
        'cosine_recall', 'adjusted_recall',
        'cosine_train_time', 'adjusted_train_time',
        'cosine_pred_time', 'adjusted_pred_time'
    ]
    df = df[cols_order]

    print("\n\n📊 RISULTATI COMPLETI")
    print(df[["embedding_model", "normalized", "cosine_accuracy", "adjusted_accuracy", "cosine_f1", "adjusted_f1"]])

    best_cosine = df.loc[df['cosine_accuracy'].idxmax()]
    best_adjusted = df.loc[df['adjusted_accuracy'].idxmax()]
    print("\n🏆 Migliori configurazioni:")
    print(f"  Cosine:   {best_cosine['embedding_model']} ({'Normalized' if best_cosine['normalized'] else 'Standard'}) "
          f"- Acc: {best_cosine['cosine_accuracy']:.4f}")
    print(f"  Adjusted: {best_adjusted['embedding_model']} ({'Normalized' if best_adjusted['normalized'] else 'Standard'}) "
          f"- Acc: {best_adjusted['adjusted_accuracy']:.4f}")

    df.to_csv('knn_comparison_results_no_leakage.csv', index=False)
    print("\n✅ Risultati salvati in 'knn_comparison_results_no_leakage.csv'")
    return df


# =======================================================
# 🔹 PIPELINE PRINCIPALE
# =======================================================
def main():
    print("="*90)
    print("PIPELINE COMPARAZIONE KNN: EMBEDDING MODELS × NORMALIZZAZIONE × METRICHE")
    print("="*90)

    N_SAMPLES = 500
    TEST_SIZE = 0.3
    K_NEIGHBORS = 5
    EMBEDDING_MODELS = [
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L6-v2",
        "paraphrase-albert-small-v2"
    ]

    texts, labels = load_and_prepare_imdb(N_SAMPLES)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
    )

    print(f"\nTrain samples: {len(X_train_text)}")
    print(f"Test samples:  {len(X_test_text)}")

    all_results = []

    for model_name in EMBEDDING_MODELS:
        print(f"\n{'='*90}\nMODELLO EMBEDDING: {model_name}\n{'='*90}")
        try:
            load_start = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - load_start

            X_train_emb, enc_train_time = generate_embeddings(model, X_train_text)
            X_test_emb, enc_test_time = generate_embeddings(model, X_test_text)
            total_encoding_time = enc_train_time + enc_test_time + load_time

            # Esperimento 1: standard embeddings
            results_std = run_experiments_on_embeddings(
                X_train_emb, X_test_emb, y_train, y_test,
                embedding_model=model_name, is_normalized=False,
                encoding_time=total_encoding_time, k_neighbors=K_NEIGHBORS
            )
            all_results.append(results_std)

            # Esperimento 2: embeddings normalizzati (solo con stats del train)
            print("\n  🧮 Normalizzazione embeddings (Z-score su TRAIN)...")
            X_train_norm, X_test_norm = normalize_embeddings_by_train(X_train_emb, X_test_emb)

            results_norm = run_experiments_on_embeddings(
                X_train_norm, X_test_norm, y_train, y_test,
                embedding_model=model_name, is_normalized=True,
                encoding_time=total_encoding_time, k_neighbors=K_NEIGHBORS
            )
            all_results.append(results_norm)

        except Exception as e:
            print(f"❌ Errore con modello {model_name}: {e}")
            continue

    if all_results:
        create_comparison_table(all_results)
    else:
        print("\n❌ Nessun risultato generato.")

# =======================================================
if __name__ == "__main__":
    main()
