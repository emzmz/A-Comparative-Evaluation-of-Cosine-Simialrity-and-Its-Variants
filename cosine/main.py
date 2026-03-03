import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from metrics_norm import adjusted_cosine_similarity, isc_similarity, sqrt_cosine_similarity, min_max_normalize
from load_data import load_data
from embeddings import compute_or_load_word_embeddings
from evaluation import evaluate_similarity_metrics

def main():
    # Crea le directory necessarie se non esistono
    os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
    os.makedirs(config.INTERIM_DATA_PATH, exist_ok=True)
    
    # Carica i dataset di benchmark
    print("Caricamento dei dataset...")
    datasets = load_data()
    
    # Definisci i modelli di embedding da valutare
    models = [
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
    
    # Definisci tutte le metriche di similarità
    # NOTA: Tutte le metriche ora riceveranno embedding normalizzati localmente
    similarity_metrics = {
        "cosine":  lambda emb1, emb2: np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)),
        "adjusted_cosine": adjusted_cosine_similarity
    }
    
    # Prepara i risultati in formato tabellare
    all_results = []
    
    # Estrai tutte le parole uniche dai dataset
    unique_words = set()
    for dataset_name, dataset in datasets.items():
        unique_words.update(dataset['word1'].tolist())
        unique_words.update(dataset['word2'].tolist())
    
    print(f"Totale parole uniche: {len(unique_words)}")
    
    # Valuta ogni combinazione di modello, metrica e dataset
    print("Valutazione delle metriche di similarità...")
    for model_name in tqdm(models, desc="Modelli"):
        # Calcoliamo o carichiamo gli embedding per tutte le parole uniche
        print(f"\nCalcolo o caricamento degli embedding per il modello: {model_name}")
        model_embeddings = {}
        not_found_words = []
        
        for word in tqdm(unique_words, desc="Parole", leave=False):
            embedding = compute_or_load_word_embeddings(word, model_name)
            if embedding is not None:
                model_embeddings[word] = embedding
            else:
                # Teniamo traccia delle parole che non hanno embedding
                not_found_words.append(word)
        
        # Stampa il numero di parole senza embedding e alcune di esse come esempio
        if not_found_words:
            print(f"  Embedding non trovati per {len(not_found_words)} parole nel modello {model_name}")
            print(f"  Esempi di parole non trovate: {not_found_words[:5]}")
        
        # NOTA: NON facciamo più normalizzazione globale qui
        # La normalizzazione min-max sarà applicata localmente in evaluation.py
        
        for metric_name, metric_func in tqdm(similarity_metrics.items(), desc="Metriche", leave=False):
            # Crea un dizionario di metriche con solo quella corrente per valutare una alla volta
            current_metric = {metric_name: metric_func}
            
            for dataset_name, dataset in tqdm(datasets.items(), desc="Dataset", leave=False):
                # Calcola le metriche di valutazione
                # Passiamo model_embeddings (non normalizzati) e la funzione di normalizzazione
                eval_results = evaluate_similarity_metrics(
                    dataset,
                    model_name,
                    compute_or_load_word_embeddings,
                    current_metric,
                    model_embeddings,  # Passiamo gli embedding NON normalizzati
                    apply_local_normalization=True  # Flag per attivare normalizzazione locale
                )
                
                # Formatta i risultati
                result_row = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'metric': metric_name,
                    'pearson': eval_results.get(f'{metric_name}_pearson', np.nan),
                    'pearson_p': eval_results.get(f'{metric_name}_pearson_p', np.nan),
                    'spearman': eval_results.get(f'{metric_name}_spearman', np.nan),
                    'spearman_p': eval_results.get(f'{metric_name}_spearman_p', np.nan),
                    'valid_pairs': eval_results.get('valid_pairs', 0),
                    'total_pairs': eval_results.get('total_pairs', 0)
                }
                all_results.append(result_row)
    
    # Converti in DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Salva i risultati in CSV
    output_path = os.path.join(config.INTERIM_DATA_PATH, "similarity_evaluation_locale.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nRisultati salvati in: {output_path}")
    return results_df

if __name__ == "__main__":
    main()