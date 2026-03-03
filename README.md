# Thesis Project

## A Comparative Evaluation of Cosine Similarity and Its Variants

### Overview

This project presents a systematic benchmarking of four semantic similarity measures within different embedding spaces. The objective is to evaluate the effectiveness of cosine-based similarity metrics in both intrinsic and extrinsic experimental settings.

The study compares:

* Cosine Similarity
* Square-Root Cosine Similarity
* Adjusted Cosine Similarity
* Improved Square-Root Cosine Similarity

---

### Embedding Models

The evaluation includes both contextual and static embeddings.

Transformer-based models:

* all-mpnet-base-v2
* all-distilroberta-v1
* all-MiniLM-L12-v2
* paraphrase-MiniLM-L6-v2
* paraphrase-albert-small-v2

Static embedding models:

* Word2Vec
* FastText
* GloVe

This selection enables comparison across contextualized sentence-transformer architectures and traditional distributional word representations.

---

### Motivation

Semantic similarity is central to numerous NLP tasks. While cosine similarity is widely adopted, several variants have been proposed to address limitations related to bias, scale sensitivity, and embedding distribution effects. This project investigates whether such variants offer measurable improvements over standard cosine similarity.

---

### Normalization Strategy

To ensure all similarity metrics are computable and comparable, Min–Max normalization is applied.

Two normalization strategies are explored:

* Global Min–Max normalization
* Local Min–Max normalization

Global normalization preserves the overall vector structure and distorts relative distances less than local normalization.
---

### Intrinsic Evaluation

The intrinsic assessment measures correlation between the metrics similarity scores and human judgments.

* Four standard benchmark datasets are used.
* Each dataset contains pairs of words with associated human similarity scores.
* Metric outputs are compared against human annotations using:

  * Pearson correlation coefficient
  * Spearman rank correlation coefficient

Results indicate:

* Cosine similarity is the strongest overall performer.
* Adjusted cosine similarity achieves competitive and promising results.

---

### Extrinsic Evaluation

The extrinsic experiment evaluates similarity measures within a downstream classification task.

* The following machine learning algorithms are evaluated:

     * k-Nearest Neighbors (KNN)
     * Logistic Regression
     * Support Vector Machines (SVM)
     * Random Forest
     * Decision Tree
     * Naive Bayes
       
* Similarity scores serve as features within the classification framework.
* Performance is evaluated comparatively across metrics and embedding spaces.

Results show:

* Adjusted cosine similarity performs best in the normalized embedding space.
* It consistently outperforms other variants in the classification setting.

---

### Project Structure

* Data loading and preprocessing
* Embedding computation and caching
* Local normalization pipeline
* Similarity metric implementation
* Correlation-based intrinsic evaluation
* Classification-based extrinsic evaluation
* Results export in tabular format

---

### Key Findings

1. Standard cosine similarity remains a strong and reliable baseline.
2. Adjusted cosine similarity demonstrates robust performance, particularly in normalized spaces.
3. Normalization strategy significantly impacts metric behavior.
4. Metric performance varies depending on whether evaluation is intrinsic or task-driven.

---

### Conclusion

This project provides an empirical comparison of cosine similarity variants across multiple embedding paradigms and evaluation settings. The results highlight the continued relevance of cosine similarity while identifying adjusted cosine as a competitive alternative, especially in extrinsic tasks involving normalized embedding spaces.

---
