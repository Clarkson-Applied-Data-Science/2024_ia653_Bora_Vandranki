# 2024_ia653_Bora_Vandranki

# Skin Disease Text Classification Pipeline

This README provides a comprehensive, step-by-step explanation of the workflow and rationale behind a skin disease text classification project. The project involves reading textual descriptions of various skin diseases, preprocessing and transforming the data, and training multiple machine learning and deep learning models to classify these descriptions into their correct disease categories. Additionally, we explore data augmentation, hyperparameter tuning, transformer models, zero-shot classification, and thorough evaluation procedures.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Environment and Setup](#environment-and-setup)
4. [Data Loading and Exploration](#data-loading-and-exploration)
5. [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
6. [Baseline Models](#baseline-models)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Model Evaluation Metrics](#model-evaluation-metrics)
9. [Data Augmentation](#data-augmentation)
10. [Evaluating Models on Augmented Data](#evaluating-models-on-augmented-data)
11. [Neural Network Models](#neural-network-models)
12. [Transformers (DistilBERT)](#transformers-distilbert)
13. [Zero-Shot Classification](#zero-shot-classification)
14. [Confusion Matrices and Classification Reports](#confusion-matrices-and-classification-reports)
15. [Visualization of Results](#visualization-of-results)
16. [Future Work](#future-work)
17. [License](#license)

---

## Project Overview
This project aims to classify a set of textual descriptions into their corresponding skin disease categories. By experimenting with various techniques—from classic machine learning methods to deep neural networks and transformer-based language models—we seek to determine which approaches yield the best predictive performance.

Key aspects of this project include:

- Comparing baseline models (Naive Bayes, Logistic Regression) with more sophisticated methods (Neural Networks, RNNs, Transformers).
- Employing data augmentation to improve model robustness and generalization.
- Leveraging hyperparameter tuning and zero-shot classification to push performance further.
- Visualizing results through confusion matrices and classification reports that use actual disease names for better interpretability.

---

## Dataset Description
We start with a dataset containing rows of text descriptions paired with a labeled disease category. The textual data typically includes symptom descriptions and other dermatological details. From this dataset, we aim to classify the texts into one of several skin disease categories.

Key points about the dataset:
- Each row contains a disease name (the target) and a textual description.
- There are multiple disease types, and we may have an imbalanced class distribution where some diseases have more samples than others.
- The dataset can be split into training and testing subsets to assess model generalization.

---

## Environment and Setup
To replicate the work described here, ensure the following environment setup:
- A modern version of Python (3.x) and essential libraries for data science (e.g., pandas, NumPy).
- Libraries for machine learning and deep learning, such as scikit-learn for classical models and TensorFlow or PyTorch for neural networks.
- The Hugging Face `transformers` library and related `datasets` and `tokenizers` packages for working with transformer-based models.
- If augmentation is performed, make sure NLTK and WordNet corpus are available for synonym extraction.

---

## Data Loading and Exploration
The process begins by reading the dataset from a CSV file into a data structure (like a pandas DataFrame). After loading, we:
- Display the first few samples to understand the data format.
- Count the number of observations and unique disease classes.
- Determine how frequently each disease occurs to identify potential class imbalance.
- Compute descriptive statistics such as the average length of the text descriptions.

Through these steps, we gain insights into the dataset’s structure and potential preprocessing needs.

---

## Preprocessing and Feature Engineering
Before training models, we need to transform the raw text into numerical representations:

1. **Text Cleaning:**  
   This may include converting to lowercase, removing punctuation, and stripping extra whitespace. A cleaner input often makes downstream models more robust.

2. **Stop Word Removal:**  
   Removing common words that do not convey specific meaning can improve model performance, especially for linear models.

3. **TF-IDF Vectorization for Classical Models:**  
   Converting raw text into TF-IDF vectors provides a convenient numerical representation. TF-IDF captures both the frequency of words and their importance relative to other documents.

4. **Label Encoding:**  
   Converting disease names into numeric form allows models to handle the classes during training and evaluation.

---

## Baseline Models
Two straightforward baseline models are trained first:

- **Naive Bayes:**  
  A simple probabilistic model often used in text classification, known for its speed and surprisingly strong performance on text data.
  
- **Logistic Regression:**  
  A linear model that often outperforms Naive Bayes in many text classification tasks, providing a solid baseline to improve upon.

By comparing these baselines, we establish a performance benchmark to gauge the impact of more advanced techniques later.

---

## Hyperparameter Tuning
Instead of relying on default parameters, we employ systematic searches (e.g., grid search) to find the best hyperparameters for the baseline models. Hyperparameter tuning can substantially improve performance by identifying optimal regularization strengths, smoothing parameters, or other model-specific settings.

Tuning steps include:
- Defining a parameter grid for each model.
- Running cross-validation to evaluate parameters.
- Selecting the parameter set that maximizes validation accuracy.

---

## Model Evaluation Metrics
To measure how well each model performs, we consider multiple metrics:

- **Accuracy:**  
  The proportion of correct predictions. While intuitive, it might not fully capture performance if classes are imbalanced.
  
- **Precision, Recall, and F1-Score (via Classification Reports):**  
  These metrics provide a more detailed breakdown of performance on each class. Precision and recall highlight how well a model identifies each disease correctly, while F1 balances both.

- **Confusion Matrix:**  
  A matrix that illustrates where the model makes mistakes, showing misclassifications by mapping predicted classes against true classes.

---

## Data Augmentation
To improve model generalization, we can increase training data diversity:

- **Synonym Replacement:**  
  For each text description, some words can be replaced with synonyms drawn from a lexical database like WordNet. This creates slightly varied training examples that can help the model become more robust against variations in input phrasing.

This augmentation produces an expanded dataset, potentially improving the classifier’s ability to handle unseen examples.

---

## Evaluating Models on Augmented Data
After augmentation, we train and evaluate models again:

- Compare performance metrics before and after augmentation.
- Look for improvements in accuracy, recall, and precision on augmented test sets.
- Check if the model overcomes previous weaknesses by reducing error rates on particular classes.

---

## Neural Network Models
Going beyond linear models, we introduce neural networks:

1. **Fully Connected Neural Network (NN):**  
   - Convert texts into sequences of integers (tokens).
   - Use an embedding layer to represent words in a dense vector space.
   - Pass the embeddings through multiple dense layers.
   - Aim for improved feature learning compared to fixed TF-IDF vectors.

2. **RNN / LSTM Model:**  
   - Employ recurrent layers such as LSTM to capture sequential dependencies in text.
   - LSTM networks can handle long-term dependencies better than simple feedforward networks, potentially improving classification of more complex disease descriptions.

---

## Transformers (DistilBERT)
We leverage modern language models pre-trained on large corpora:

- **DistilBERT:**  
  A lightweight version of BERT trained on a vast text corpus. It’s fine-tuned on our dataset for classification:
  
  Advantages:
  - The model already knows a lot about language structure and common patterns.
  - Often yields strong improvements over traditional approaches due to contextual embeddings.

---

## Zero-Shot Classification
To test the model’s capability without any direct training on the dataset’s specific classes, we employ a zero-shot approach:

- **BART Large MNLI Model:**  
  By using a zero-shot classification pipeline, we can present any text and a set of candidate disease names. The model tries to match the text to the most likely class without having seen any training examples.  
  This approach demonstrates how well general-purpose language models can handle domain-specific classification tasks.

---

## Confusion Matrices and Classification Reports
To improve interpretability:

- **Textual Class Labels in Reports:**  
  Instead of numeric labels, we present each metric under the corresponding disease name. This helps domain experts and stakeholders understand where models struggle.

- **Labeled Confusion Matrices:**  
  Annotating both axes of the confusion matrices with disease names makes it easier to see which diseases get confused. For instance, a model may frequently mistake Eczema for Psoriasis, suggesting a need for more training examples or refined preprocessing.

---

## Visualization of Results
Visual representations help interpret performance:

- **Bar Charts for Accuracy:**  
  Compare models (Naive Bayes, Logistic Regression, NN, RNN, Transformers) side by side on both original and augmented datasets.
  
- **Learning Curves:**  
  Show how models improve over epochs for neural networks. Plots of training vs. validation accuracy and loss help diagnose overfitting or underfitting.

- **Improvement Over Baselines:**  
  Visual comparisons highlight improvements gained from augmentation, advanced models, or hyperparameter tuning.

---

## Future Work
Potential enhancements to further improve the pipeline:

- **Advanced Augmentation Methods:**  
  Beyond synonym replacement, consider back-translation or contextual augmentation using language models.
  
- **Additional Transformer Models:**  
  Experiment with larger or more specialized transformer models (e.g., RoBERTa, BioBERT) trained on medical texts.
  
- **Explainability Tools:**  
  Apply methods like LIME or SHAP to understand why models make certain predictions, potentially yielding insights into model behavior and biases.

- **Domain-Specific Embeddings:**  
  Further fine-tune embeddings with a large corpus of dermatological literature to achieve better domain adaptation.

---

## License
The project is generally released under an MIT License (or a similar open-source license), allowing users to adapt and build upon the work with minimal restrictions. Please refer to the LICENSE file included in the project repository for full details.

---

By following this detailed workflow—from understanding the data and applying simple models through to sophisticated approaches like Transformers and zero-shot classification—you gain a broad perspective on effective text classification strategies for the domain of skin disease descriptions.
