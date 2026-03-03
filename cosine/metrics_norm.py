import numpy as np

def min_max_normalize(vector):
    """
    Normalizza un vettore usando min-max scaling LOCALE nell'intervallo [0, 1].
    Ogni vettore viene normalizzato indipendentemente usando i propri valori min/max.
    
    Args:
        vector: numpy array da normalizzare
        
    Returns:
        numpy array normalizzato
    """
    v_min = np.min(vector)
    v_max = np.max(vector)
    
    # Evita divisione per zero se tutti i valori sono uguali
    if v_max == v_min:
        return np.ones_like(vector) * 0.5  # Valore neutro
    
    return (vector - v_min) / (v_max - v_min)

def adjusted_cosine_similarity(embedding1, embedding2):
    """
    Calcola la adjusted cosine similarity tra due vettori di embedding.
    NOTA: Si assume che gli embedding siano già normalizzati min-max localmente.

    Parameters:
        embedding1 (np.ndarray): vettore per la parola 1 (normalizzato)
        embedding2 (np.ndarray): vettore per la parola 2 (normalizzato)

    Returns:
        float: valore della adjusted cosine similarity
    """
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()

    # Sottraiamo la media
    emb1_adjusted = emb1 - np.mean(emb1)
    emb2_adjusted = emb2 - np.mean(emb2)
    
    # Calcolo della similarità
    numerator = np.dot(emb1_adjusted, emb2_adjusted)
    denominator = np.linalg.norm(emb1_adjusted) * np.linalg.norm(emb2_adjusted)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def sqrt_cosine_similarity(embedding1, embedding2):
    """
    Calcola la sqrt-cosine similarity tra due vettori di embedding.
    NOTA: Si assume che gli embedding siano già normalizzati min-max localmente.
    
    Formula: SqrtCos(x, y) = sum(sqrt(x_i * y_i)) / (sum(x_i) * sum(y_i))
    
    Args:
        embedding1: Primo vettore di embedding (già normalizzato)
        embedding2: Secondo vettore di embedding (già normalizzato)
        
    Returns:
        float: Valore di similarità sqrt-cosine
    """
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()
    
    # Calcola il numeratore (somma delle radici quadrate dei prodotti)
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
    NOTA: Si assume che gli embedding siano già normalizzati min-max localmente.
    
    Formula: ISC(x, y) = sum(sqrt(x_i * y_i)) / (sqrt(sum(x_i)) * sqrt(sum(y_i)))
    
    Args:
        embedding1: Primo vettore di embedding (già normalizzato)
        embedding2: Secondo vettore di embedding (già normalizzato)
        
    Returns:
        float: Valore di similarità ISC
    """
    emb1 = np.array(embedding1).flatten()
    emb2 = np.array(embedding2).flatten()
    
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