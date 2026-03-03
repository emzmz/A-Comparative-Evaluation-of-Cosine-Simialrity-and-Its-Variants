import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. Load pre-trained model
print("Loading pre-trained model...")
model = api.load("word2vec-google-news-300")  # approximately 1.6 GB

# 2. Define words per category (based on your image)
categories = {
    "Royalty": ["man", "woman", "king", "queen", "prince", "princess"],
    "Fruits": ["grape", "apple", "orange", "mango", "banana"],
    "Animals": ["mouse", "cat", "dog", "lion", "tiger", "elephant"],
    "Transport": ["train", "bus", "bicycle", "car", "plane"],
    "Cities": ["berlin", "paris", "tokyo", "london"],
    "Technology": ["java", "software", "computer", "internet", "python"]
}

# 3. Prepare list of words and corresponding categories
words = []
word_categories = []
for cat, word_list in categories.items():
    for word in word_list:
        if word in model.key_to_index:
            words.append(word)
            word_categories.append(cat)

# 4. Extract embeddings and convert to NumPy array
embeddings = np.array([model[word] for word in words])

# 5. Dimensionality reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)

# 6. Colors per category
colors = {
    "Royalty": "red",
    "Fruits": "orange",
    "Animals": "green",
    "Transport": "blue",
    "Cities": "purple",
    "Technology": "brown"
}

# 7. Plot
plt.figure(figsize=(12, 6))
for word, coord, cat in zip(words, embeddings_2d, word_categories):
    plt.scatter(coord[0], coord[1], color=colors[cat], s=100, alpha=0.8)
    plt.text(coord[0]+0.3, coord[1]+0.3, word, fontsize=10)

# Legend
for cat, color in colors.items():
    plt.scatter([], [], color=color, label=cat, s=100)
plt.legend(fontsize=12, loc='lower left')
plt.xlabel("X Axis", fontsize=11)
plt.ylabel("Y Axis", fontsize=11)
plt.tight_layout()
plt.show()