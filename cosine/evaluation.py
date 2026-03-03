import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def evaluate_similarity_metrics(dataset, model_name, compute_embeddings_fn, metrics_dict, compute_or_load_word_embeddings=None):
    """
    Valuta diverse metriche di similarità su un dataset usando gli embedding di un modello.
    
    Args:
        dataset (pd.DataFrame): DataFrame con colonne ['word1', 'word2', 'human_score']
        model_name (str): Nome del modello di embedding
        compute_embeddings_fn (function): Funzione per calcolare gli embedding
        metrics_dict (dict): Dizionario di funzioni per calcolare diverse metriche di similarità
        min_max_normalized_embeddings (dict, optional): Dizionario con embedding già normalizzati min-max
    
    Returns:
        dict: Dizionario con risultati di correlazione per ogni metrica
    """
    words1 = dataset['word1'].tolist()
    words2 = dataset['word2'].tolist()
    human_scores = dataset['human_score'].tolist()
    
    # Calcola gli embedding per ogni parola
    embeddings1 = []
    embeddings2 = []
    valid_pairs = 0
    total_pairs = len(words1)
    
    for word1, word2 in zip(words1, words2):
        # Se abbiamo già gli embedding normalizzati min-max, li usiamo
        if compute_or_load_word_embeddings is not None:
            emb1 = compute_or_load_word_embeddings.get(word1)
            emb2 = compute_or_load_word_embeddings.get(word2)
        else:
            # Altrimenti, calcoliamo gli embedding normali
            emb1 = compute_embeddings_fn(word1, model_name)
            emb2 = compute_embeddings_fn(word2, model_name)
        
        if emb1 is not None and emb2 is not None:
            # Normalizza gli embedding per la norma L2 se richiesto (per cosine similarity)
            embeddings1.append(emb1)
            embeddings2.append(emb2)
            valid_pairs += 1
        else:
            # Se un embedding non è disponibile, salta questa coppia
            embeddings1.append(None)
            embeddings2.append(None)
    
    results = {
        'valid_pairs': valid_pairs,
        'total_pairs': total_pairs
    }
    
    # Calcola le similarità per ogni metrica
    for metric_name, metric_fn in metrics_dict.items():
        metric_scores = []
        valid_human_scores = []
        
        for i in range(total_pairs):
            if embeddings1[i] is not None and embeddings2[i] is not None:
                try:
                    sim_score = metric_fn(embeddings1[i], embeddings2[i])
                    if not np.isnan(sim_score):
                        metric_scores.append(sim_score)
                        valid_human_scores.append(human_scores[i])
                except Exception as e:
                    # In caso di errore durante il calcolo della metrica, continua con la coppia successiva
                    pass
        
        # Calcola le correlazioni di Pearson e Spearman
        if len(metric_scores) > 1:
            pearson_corr, pearson_p = pearsonr(metric_scores, valid_human_scores)
            spearman_corr, spearman_p = spearmanr(metric_scores, valid_human_scores)
            
            results[f'{metric_name}_pearson'] = pearson_corr
            results[f'{metric_name}_pearson_p'] = pearson_p
            
            results[f'{metric_name}_spearman'] = spearman_corr
            results[f'{metric_name}_spearman_p'] = spearman_p
        else:
            results[f'{metric_name}_pearson'] = np.nan
            results[f'{metric_name}_pearson_p'] = np.nan
            results[f'{metric_name}_spearman'] = np.nan
            results[f'{metric_name}_spearman_p'] = np.nan
    
    return results