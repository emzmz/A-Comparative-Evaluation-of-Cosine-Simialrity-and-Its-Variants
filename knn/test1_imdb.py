import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import time
import pandas as pd

class AdjustedCosineKNN:
    """KNN con Normalised Adjusted Cosine Similarity"""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.mean_features = None
        
    def fit(self, X, y):
        """Addestra il modello"""
        self.X_train = X
        self.y_train = np.array(y)
        # Calcola la media di ogni feature
        self.mean_features = np.mean(self.X_train, axis=0)
        return self
    
    def _adjusted_cosine_similarity(self, u, v):
        """Calcola la normalised adjusted cosine similarity"""
        # Centra i vettori sottraendo la media
        u_centered = u - self.mean_features
        v_centered = v - self.mean_features
        
        # Calcola la similarità del coseno sui vettori centrati
        dot_product = np.dot(u_centered, v_centered)
        norm_u = np.linalg.norm(u_centered)
        norm_v = np.linalg.norm(v_centered)
        
        if norm_u == 0 or norm_v == 0:
            return 0
        
        similarity = dot_product / (norm_u * norm_v)
        return similarity
    
    def predict(self, X):
        """Predice le etichette per i dati di test"""
        predictions = []
        
        for test_sample in X:
            # Calcola le similarità con tutti i campioni di training
            similarities = []
            for train_sample in self.X_train:
                sim = self._adjusted_cosine_similarity(test_sample, train_sample)
                similarities.append(sim)
            
            # Trova i k vicini più simili
            similarities = np.array(similarities)
            k_indices = np.argsort(similarities)[-self.n_neighbors:]
            
            # Voto maggioritario
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        
        return np.array(predictions)

def load_and_prepare_imdb(n_samples=2000):
    """Carica e prepara il dataset IMDB"""
    print("Caricamento dataset IMDB...")
    dataset = load_dataset('imdb', split='train')
    
    # Prendi un sottoinsieme per velocizzare
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    
    texts = dataset['text']
    labels = dataset['label']
    
    return texts, labels

def generate_embeddings(texts, model_name):
    """Genera embeddings usando un modello sentence-transformers"""
    print(f"\nGenerazione embeddings con {model_name}...")
    model = SentenceTransformer(model_name)
    
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    encoding_time = time.time() - start_time
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Tempo encoding: {encoding_time:.2f}s")
    
    return embeddings, encoding_time

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Valuta un modello e restituisce le metriche"""
    print(f"  Valutazione: {model_name}...", end=" ")
    
    # Training
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    # Prediction
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    
    print(f"✓ (Acc: {accuracy:.4f}, F1: {f1:.4f})")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'pred_time': pred_time
    }

def run_experiments(X_train, X_test, y_train, y_test, embedding_model, encoding_time, k_neighbors=5):
    """Esegue esperimenti con entrambi i modelli KNN"""
    print(f"\n{'='*70}")
    print(f"Esperimenti con embedding: {embedding_model}")
    print(f"{'='*70}")
    
    results = {
        'embedding_model': embedding_model,
        'encoding_time': encoding_time,
        'embedding_dim': X_train.shape[1]
    }
    
    # Modello 1: KNN con Cosine Similarity
    knn_cosine = KNeighborsClassifier(
        n_neighbors=k_neighbors, 
        metric='cosine',
        algorithm='brute'
    )
    
    cosine_metrics = evaluate_model(
        knn_cosine, X_train, X_test, y_train, y_test,
        "KNN Cosine"
    )
    
    for key, value in cosine_metrics.items():
        results[f'cosine_{key}'] = value
    
    # Modello 2: KNN con Adjusted Cosine Similarity
    knn_adjusted = AdjustedCosineKNN(n_neighbors=k_neighbors)
    
    adjusted_metrics = evaluate_model(
        knn_adjusted, X_train, X_test, y_train, y_test,
        "KNN Adjusted Cosine"
    )
    
    for key, value in adjusted_metrics.items():
        results[f'adjusted_{key}'] = value
    
    return results

def create_comparison_table(all_results):
    """Crea una tabella comparativa dei risultati"""
    print(f"\n{'='*70}")
    print("TABELLA COMPARATIVA COMPLETA")
    print(f"{'='*70}\n")
    
    # Crea DataFrame
    df = pd.DataFrame(all_results)
    
    # Riordina colonne
    cols_order = ['embedding_model', 'embedding_dim', 'encoding_time',
                  'cosine_accuracy', 'adjusted_accuracy',
                  'cosine_f1', 'adjusted_f1',
                  'cosine_precision', 'adjusted_precision',
                  'cosine_recall', 'adjusted_recall',
                  'cosine_train_time', 'adjusted_train_time',
                  'cosine_pred_time', 'adjusted_pred_time']
    
    df = df[cols_order]
    
    # Stampa con formattazione
    print("ACCURACY")
    print("-" * 70)
    for _, row in df.iterrows():
        model = row['embedding_model']
        cos_acc = row['cosine_accuracy']
        adj_acc = row['adjusted_accuracy']
        winner = "Cosine" if cos_acc > adj_acc else "Adjusted" if adj_acc > cos_acc else "Tie"
        print(f"{model:<30} | Cos: {cos_acc:.4f} | Adj: {adj_acc:.4f} | Winner: {winner}")
    
    print("\n\nF1-SCORE")
    print("-" * 70)
    for _, row in df.iterrows():
        model = row['embedding_model']
        cos_f1 = row['cosine_f1']
        adj_f1 = row['adjusted_f1']
        winner = "Cosine" if cos_f1 > adj_f1 else "Adjusted" if adj_f1 > cos_f1 else "Tie"
        print(f"{model:<30} | Cos: {cos_f1:.4f} | Adj: {adj_f1:.4f} | Winner: {winner}")
    
    print("\n\nTEMPI DI PREDIZIONE (secondi)")
    print("-" * 70)
    for _, row in df.iterrows():
        model = row['embedding_model']
        cos_time = row['cosine_pred_time']
        adj_time = row['adjusted_pred_time']
        faster = "Cosine" if cos_time < adj_time else "Adjusted"
        print(f"{model:<30} | Cos: {cos_time:>6.2f}s | Adj: {adj_time:>6.2f}s | Faster: {faster}")
    
    # Best overall
    print("\n\nMIGLIORI MODELLI PER METRICA")
    print("=" * 70)
    best_acc_idx = df['cosine_accuracy'].idxmax()
    best_adj_acc_idx = df['adjusted_accuracy'].idxmax()
    
    print(f"Miglior Accuracy (Cosine):          {df.loc[best_acc_idx, 'embedding_model']:<30} ({df.loc[best_acc_idx, 'cosine_accuracy']:.4f})")
    print(f"Miglior Accuracy (Adjusted Cosine): {df.loc[best_adj_acc_idx, 'embedding_model']:<30} ({df.loc[best_adj_acc_idx, 'adjusted_accuracy']:.4f})")
    
    best_f1_idx = df['cosine_f1'].idxmax()
    best_adj_f1_idx = df['adjusted_f1'].idxmax()
    
    print(f"Miglior F1 (Cosine):                {df.loc[best_f1_idx, 'embedding_model']:<30} ({df.loc[best_f1_idx, 'cosine_f1']:.4f})")
    print(f"Miglior F1 (Adjusted Cosine):       {df.loc[best_adj_f1_idx, 'embedding_model']:<30} ({df.loc[best_adj_f1_idx, 'adjusted_f1']:.4f})")
    
    return df

def main():
    """Pipeline principale"""
    print("="*70)
    print("PIPELINE COMPARAZIONE KNN CON MULTIPLI EMBEDDING MODELS")
    print("="*70)
    
    # Parametri
    N_SAMPLES = 2000  # Aumenta per risultati più robusti
    TEST_SIZE = 0.3
    K_NEIGHBORS = 5
    
    # Modelli di embedding da testare
    EMBEDDING_MODELS = [
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L6-v2",
        "paraphrase-albert-small-v2"
    ]
    
    # 1. Carica dati
    texts, labels = load_and_prepare_imdb(N_SAMPLES)
    
    # 2. Split train/test (sui testi, prima degli embeddings)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    
    print(f"\nTrain samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")
    
    # 3. Loop su tutti i modelli di embedding
    all_results = []
    
    for embedding_model in EMBEDDING_MODELS:
        try:
            # Genera embeddings
            X_train_emb, enc_time_train = generate_embeddings(X_train_text, embedding_model)
            X_test_emb, enc_time_test = generate_embeddings(X_test_text, embedding_model)
            
            total_enc_time = enc_time_train + enc_time_test
            
            # Esegui esperimenti con entrambi i KNN
            results = run_experiments(
                X_train_emb, X_test_emb, y_train, y_test,
                embedding_model, total_enc_time, K_NEIGHBORS
            )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"\n❌ Errore con {embedding_model}: {str(e)}")
            continue
    
    # 4. Crea tabella comparativa finale
    if all_results:
        df_results = create_comparison_table(all_results)
        
        # Salva risultati in CSV
        df_results.to_csv('knn_comparison_results.csv', index=False)
        print(f"\n✓ Risultati salvati in 'knn_comparison_results.csv'")
    else:
        print("\n❌ Nessun risultato da visualizzare")

if __name__ == "__main__":
    main()