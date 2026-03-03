import gensim.downloader as api
import numpy as np

def test_word_availability():
    """
    Testa la disponibilità di parole specifiche nei modelli word2vec, fasttext e glove.
    """
    # Parole da testare (incluse quelle mancanti dal log)
    test_words = [
        # Parole dal log che risultano mancanti
        'grey', 'colour', 'theatre', 'harbour',
        'Maradona', 'Palestinian', 'Mexico', 'CD', 'Jerusalem',
        
        # Alcune varianti britanniche vs americane
        'color', 'theater', 'harbor',
        
        # Parole comuni per confronto
        'cat', 'dog', 'house', 'computer',
        
        # Nomi propri
        'America', 'England', 'Paris'
    ]
    
    models_config = {
        'word2vec': 'word2vec-google-news-300',
        'fasttext': 'fasttext-wiki-news-subwords-300',
        'glove': 'glove-wiki-gigaword-100'
    }
    
    print("="*80)
    print("TEST DISPONIBILITÀ PAROLE NEI MODELLI DI EMBEDDING")
    print("="*80)
    
    results = {}
    
    for model_name, model_path in models_config.items():
        print(f"\n{'='*80}")
        print(f"Caricamento modello: {model_name}")
        print(f"{'='*80}")
        
        try:
            model = api.load(model_path)
            print(f"✓ Modello caricato con successo")
            print(f"  Dimensione vocabolario: {len(model.key_to_index):,} parole")
            print(f"  Dimensione embedding: {model.vector_size}")
            
            results[model_name] = {
                'found': [],
                'not_found': [],
                'model': model
            }
            
            print(f"\n{'-'*80}")
            print(f"Test parole:")
            print(f"{'-'*80}")
            
            for word in test_words:
                if word in model:
                    results[model_name]['found'].append(word)
                    embedding = model[word]
                    print(f"  ✓ '{word}' - TROVATA (embedding shape: {embedding.shape})")
                else:
                    results[model_name]['not_found'].append(word)
                    print(f"  ✗ '{word}' - NON TROVATA")
            
            # Per fasttext, che supporta subword, testiamo anche parole OOV
            if model_name == 'fasttext':
                print(f"\n{'-'*40}")
                print(f"Test FastText con parole inventate (grazie ai subword):")
                print(f"{'-'*40}")
                fake_words = ['impossibleword123', 'xyzzz', 'qwerty']
                for word in fake_words:
                    try:
                        embedding = model[word]
                        print(f"  ✓ '{word}' - Generato embedding (shape: {embedding.shape})")
                    except:
                        print(f"  ✗ '{word}' - Impossibile generare")
            
        except Exception as e:
            print(f"✗ Errore nel caricamento del modello: {e}")
            results[model_name] = None
    
    # Riepilogo finale
    print(f"\n{'='*80}")
    print("RIEPILOGO FINALE")
    print(f"{'='*80}")
    
    for model_name in models_config.keys():
        if results[model_name] is not None:
            found_count = len(results[model_name]['found'])
            not_found_count = len(results[model_name]['not_found'])
            total = found_count + not_found_count
            percentage = (found_count / total * 100) if total > 0 else 0
            
            print(f"\n{model_name.upper()}:")
            print(f"  Trovate: {found_count}/{total} ({percentage:.1f}%)")
            print(f"  Mancanti: {not_found_count}/{total}")
            
            if not_found_count > 0:
                print(f"  Parole mancanti: {', '.join(results[model_name]['not_found'])}")
    
    # Analisi varianti ortografiche
    print(f"\n{'='*80}")
    print("ANALISI VARIANTI ORTOGRAFICHE (UK vs US)")
    print(f"{'='*80}")
    
    variants = [
        ('grey', 'gray'),
        ('colour', 'color'),
        ('theatre', 'theater'),
        ('harbour', 'harbor')
    ]
    
    for model_name, result in results.items():
        if result is not None:
            print(f"\n{model_name.upper()}:")
            model = result['model']
            for uk_word, us_word in variants:
                uk_found = uk_word in model
                us_found = us_word in model
                print(f"  {uk_word:12s} (UK): {'✓' if uk_found else '✗'}")
                print(f"  {us_word:12s} (US): {'✓' if us_found else '✗'}")
                print()
    
    return results

if __name__ == "__main__":
    results = test_word_availability()
    
    print("\n" + "="*80)
    print("Test completato!")
    print("="*80)