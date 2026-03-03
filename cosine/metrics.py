import numpy as np

def adjusted_cosine_similarity(embedding1, embedding2):
    """
    Calcola la adjusted cosine similarity tra due vettori di embedding.

    Parameters:
        embedding1 (np.ndarray): vettore per la parola 1
        embedding2 (np.ndarray): vettore per la parola 2

    Returns:
        float: valore della adjusted cosine similarity
    """
    # Assicuriamoci che gli embedding siano array numpy e senza reshape
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()

    # Sottraiamo la media
    emb1_adjusted = emb1 - np.mean(emb1)
    emb2_adjusted = emb2 - np.mean(emb2)
    
    # Calcolo della similarità
    numerator = np.dot(emb1_adjusted, emb2_adjusted)
    denominator = np.linalg.norm(emb1_adjusted) * np.linalg.norm(emb2_adjusted)
    
    # Evita divisione per zero
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def sqrt_cosine_similarity(embedding1, embedding2):
    """
    Calcola la sqrt-cosine similarity originale tra due vettori di embedding.
    
    Formula: SqrtCos(x, y) = sum(sqrt(x_i * y_i)) / (sum(x_i) * sum(y_i))
    
    Args:
        x: Primo vettore di embedding
        y: Secondo vettore di embedding
        
    Returns:
        float: Valore di similarità sqrt-cosine, o NaN se ci sono valori negativi
    """
    # Assicuriamoci che gli embedding siano array numpy e senza reshape
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()
    
    # Con la normalizzazione min-max non dovremmo più avere valori negativi
    # ma facciamo il controllo per sicurezza
    if np.any(emb1 < 0) or np.any(emb2 < 0):
        return np.nan
    
    # Calcola il numeratore (somma delle radici quadrate dei prodotti)
    # Gestisci il caso in cui il prodotto è 0 (che è valido, ma sqrt(0) = 0)
    products = np.multiply(emb1, emb2)
    numerator = np.sum(np.sqrt(products))
    
    # Calcola il denominatore (prodotto delle somme)
    denominator = np.sum(emb1) * np.sum(emb2)
    
    # Evita la divisione per zero
    if denominator == 0:
        return np.nan
    
    return numerator / denominator

def isc_similarity(embedding1, embedding2):
    """
    Calcola la ISC (Improved Sqrt-Cosine) similarity tra due vettori di embedding.
    
    Formula: ISC(x, y) = sum(sqrt(x_i * y_i)) / (sqrt(sum(x_i)) * sqrt(sum(y_i)))
    
    Args:
        x: Primo vettore di embedding
        y: Secondo vettore di embedding
        
    Returns:
        float: Valore di similarità ISC, o NaN se ci sono valori negativi
    """
    # Assicuriamoci che gli embedding siano array numpy e senza reshape
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()

    # Con la normalizzazione min-max non dovremmo più avere valori negativi
    # ma facciamo il controllo per sicurezza
    if np.any(emb1 < 0) or np.any(emb2 < 0):
        return np.nan
    
    # Calcola il numeratore (somma delle radici quadrate dei prodotti)
    products = np.multiply(emb1, emb2)
    numerator = np.sum(np.sqrt(products))
    
    # Calcola il denominatore (prodotto delle radici quadrate delle somme)
    sum_x = np.sum(emb1)
    sum_y = np.sum(emb2)
    if sum_x <= 0 or sum_y <= 0:
        return np.nan
    
    denominator = np.sqrt(sum_x) * np.sqrt(sum_y)
    
    return numerator / denominator

