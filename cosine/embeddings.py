import config
import shelve
import os
import numpy as np
from sentence_transformers import SentenceTransformer


# auxiliary funciton 
def get_static_word_embedding(word, model_name="word2vec"):
    """
    Get the embedding of a word using Word2Vec, FastText, or GloVe.
    Downloads and caches the model on first use (caches via function attribute).
    """
    import gensim.downloader as api
    
    if not hasattr(get_static_word_embedding, "_cache"):
        get_static_word_embedding._cache = {}
    
    if model_name not in get_static_word_embedding._cache:
        if model_name == "word2vec":
            get_static_word_embedding._cache[model_name] = api.load("word2vec-google-news-300")
        elif model_name == "fasttext":
            get_static_word_embedding._cache[model_name] = api.load("fasttext-wiki-news-subwords-300")
        elif model_name == "glove":
            # Usando gensim invece di torchtext per GloVe
            get_static_word_embedding._cache[model_name] = api.load("glove-wiki-gigaword-100")
        else:
            raise ValueError("Model must be 'word2vec', 'fasttext', or 'glove'.")
    
    model = get_static_word_embedding._cache[model_name]
    
    if word in model:
        return model[word]
    else:
        return None

# main function
def compute_or_load_word_embeddings(word, model_name="paraphrase-MiniLM-L6-v2"):
    static_models = ['word2vec', 'fasttext', 'glove']
    if model_name not in static_models:
        # Create a unique key using both the model name and the word.
        key = f"{model_name}::{word}"
        cache_file = os.path.join(config.INTERIM_DATA_PATH, f"{model_name}_embedding_cache.db")
        
        # Open the shelve file which acts as our persistent cache.
        with shelve.open(cache_file) as embedding_cache:
            if key in embedding_cache:
                # Return the cached embedding if it exists.
                return embedding_cache[key]
            else:
                # Otherwise, load the model, compute the embedding, and cache it.
                model = SentenceTransformer(model_name)
                embedding = np.array(model.encode(word))
                embedding_cache[key] = embedding
                return embedding
    else:
        return get_static_word_embedding(word, model_name)
    

