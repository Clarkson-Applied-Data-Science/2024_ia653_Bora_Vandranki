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
17. [License](#license)  -  daat leakage 
18 refernce


---

## Project Overview
This project aims to classify a set of textual descriptions into their corresponding skin disease categories. By experimenting with various techniques—from classic machine learning methods to deep neural networks and transformer-based language models—we seek to determine which approaches yield the best predictive performance.

Key aspects of this project include:

- Comparing baseline models (Naive Bayes, Logistic Regression) with more sophisticated methods (Neural Networks, RNNs, Transformers).
- Employing data augmentation to improve model robustness and generalization.
- Leveraging hyperparameter tuning and zero-shot classification to push performance further.
- Visualizing results through confusion matrices and classification reports that use actual disease names for better interpretability.

---

## Dataset 

The dataset (`Skin_text_classifier.csv`) contains text descriptions of skin diseases paired with labeled disease categories. This data includes detailed symptom descriptions and other dermatological information tailored for text classification.

### Dataset Description
- **Source**: The dataset is sourced from Kaggle and can be accessed directly [here](https://www.kaggle.com/code/tanshihjen/nlp-task-skindiseasetextclassification?select=Skin_text_classifier.csv).
- **Content**: Each row consists of a 'Disease name' (the target variable) and a 'Text' field containing the description.

### Disease Overview
Below are examples of diseases included in the dataset along with common symptoms:

1. **Acne:** Characterized by inflamed red lesions, blackheads, and painful cysts.
2. **Athlete's Foot (Tinea Pedis):** Features dry, flaky patches, with itching, especially between the toes.
3. **Contact Dermatitis:** Involves dry, cracked skin and tightness after contact with irritants.
4. **Eczema:** Known for dry, itchy, red skin.
5. **Folliculitis:** Identified by small, hard, pus-filled bumps.
6. **Hives (Urticaria):** Presents with red, itchy welts and swelling on the skin.
7. **Impetigo:** Noted for itchy blisters that ooze and form crusts.
8. **Psoriasis:** Exhibits thick, red patches with silvery scales, mostly on joints.
9. **Ringworm (Tinea Corporis):** Appears as a red, scaly patch that is itchy.
10. **Rosacea:** Manifests as facial redness, swollen red bumps, and visible blood vessels.
11. **Scabies:** Features small, red bumps and intense itching, especially at night.
12. **Shingles (Herpes Zoster):** Characterized by a painful rash and fluid-filled blisters.
13. **Vitiligo:** Involves the loss of skin color in blotches.

This section provides a quick overview of the primary symptoms associated with each disease listed in the dataset, helping to facilitate better understanding and identification of conditions in clinical and educational settings.

---

## Environment and Setup
To replicate the work described here, ensure the following environment setup:
- A modern version of Python (3.x) and essential libraries for data science (e.g., pandas, NumPy).
- Libraries for machine learning and deep learning, such as scikit-learn for classical models and TensorFlow or PyTorch for neural networks.
- The Hugging Face `transformers` library and related `datasets` and `tokenizers` packages for working with transformer-based models.
- If augmentation is performed, make sure NLTK and WordNet corpus are available for synonym extraction.

**Weights & Biases Account:**
   - Necessary for utilizing pre-trained models and fine-tuning them. Weights & Biases (W&B) offers a developer platform with tools for training, fine-tuning, and leveraging foundation models. An account can be created and API access obtained from their website. For more information and to set up an account, visit [Weights & Biases](https://wandb.ai/site/).

This setup will prepare your environment for running the models and scripts provided in this project, ensuring compatibility and the ability to track and optimize model performance.


---

## Data Loading and Exploration
The process begins by reading the dataset from a CSV file into a data structure (like a pandas DataFrame). After loading, we:
- Display the first few samples to understand the data format.
- Count the number of observations and unique disease classes.
- Calculate the average length of the text descriptions across the dataset. This metric is useful for setting parameters in text preprocessing steps, such as padding or truncation when preparing data for machine learning models.
- If class imbalance is detected (for example, some diseases occurring much more frequently than others), this needs to be addressed in the modeling phase to ensure fair and effective model training. Techniques such as data augmentation, oversampling, or undersampling might be employed.

Through these steps, we gain insights into the dataset’s structure and potential preprocessing needs.

 <img src="/NLP_Project/nlp- 1.png" alt="NLP1">
 <img src="/NLP_Project/nlp- 2.png" alt="NLP2">
Distribution of Text Lengths Chart: The histogram shows the text lengths are mostly centered around 40 words, with a normal distribution. This suggests uniformity in text size, which is helpful for preprocessing steps like padding or truncation in model training.


---

## Preprocessing and Feature Engineering

Before training models, transforming raw text into numerical representations is crucial for machine learning algorithms to process the data effectively. Here are the steps involved:

1. **Text Cleaning:**
   Convert text to lowercase, remove punctuation, and strip extra whitespaces to standardize inputs and reduce model training complexity.

2. **Stop Word Removal:**
   Remove common words that do not convey specific meaning to improve model performance by reducing data dimensionality.

3. **TF-IDF Vectorization for Classical Models:**
   Convert raw text into TF-IDF vectors, providing a numeric representation that captures both the frequency of words and their importance relative to other documents.

4. **Label Encoding:**
   Convert disease names into numeric form to facilitate model processing, as models interpret numerical data better than text data.

### Example of TF-IDF Application

Consider two documents:

- **Doc1:** "Cat on the hot tin roof"
- **Doc2:** "Hot chocolate on a cold night"

Here's how TF-IDF treats these documents:

| Word       | Count in Doc1 | Count in Doc2 | IDF  | TF-IDF in Doc1 | TF-IDF in Doc2 |
|------------|---------------|---------------|------|----------------|----------------|
| cat        | 1             | 0             | High | High           | 0              |
| hot        | 1             | 1             | Low  | Low            | Low            |
| chocolate  | 0             | 1             | High | 0              | High           |

- **Word 'hot'**: Appears in both documents, resulting in a lower IDF (inverse document frequency) and subsequently lower TF-IDF scores because it is not unique.
- **Words 'cat' and 'chocolate'**: Appear only in one document each, resulting in higher IDF scores and hence higher TF-IDF values for the document they appear in. This indicates their greater importance in the context of their respective documents.

This table shows that TF-IDF effectively differentiates the importance of words based on their distribution across documents, which can significantly impact the performance of classification models by prioritizing more distinctive terms.

This structured approach to data preparation ensures that the text data is optimally formatted for training machine learning models, making them more efficient and accurate in predictions.

<img src="/NLP_Project/nlp- 4.png" alt="NLP3">

---

## Baseline Models

To establish a solid foundation for evaluating more advanced machine learning techniques, we start with two well-regarded baseline models known for their effectiveness in text classification:

1. **Naive Bayes:**
   - **Description:** Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It is particularly well-suited for text classification tasks where the dimensionality of the input space is high, and the features (words) can be treated as independent.
   - **Formula:**
     ```
     P(C_k | x_1, ..., x_n) = P(C_k) * Π P(x_i | C_k) / P(x_1, ..., x_n)
     ```
     Where `C_k` is the class variable, `x_1, ..., x_n` are the features, and `P(x_i | C_k)` is the likelihood of feature `i` given class `C_k`.
   - **Rationale:** Naive Bayes is chosen for its computational efficiency and surprisingly strong performance with text data, despite the simplicity of the model.

2. **Logistic Regression:**
   - **Description:** Logistic Regression is a robust statistical model that estimates probabilities using a logistic function, commonly used for binary classification tasks. Extended to multiclass classification, it often serves as a strong baseline for text data.
   - **Formula:**
     ```
     σ(z) = 1 / (1 + e^(-z))
     ```
     Where `σ` is the logistic function and `z` is the linear combination of features and their coefficients.
   - **Rationale:** Logistic Regression is favored for its ability to provide probability scores for different classes, offering a straightforward interpretation of results and good performance across a variety of classification problems.

### Establishing Benchmarks

By employing these baseline models, we aim to:
- **Set Performance Benchmarks:** Establish a performance baseline to gauge the effectiveness of more advanced models.
- **Simplicity vs. Complexity:** Understand the trade-offs between simpler algorithms and more complex models in terms of computational cost and predictive accuracy.

These models not only serve as a proof of concept but also help in identifying potential areas for improvement as we explore more sophisticated machine learning techniques later in the project.


---

## Hyperparameter Tuning

Hyperparameter tuning is a critical step in enhancing the performance of our baseline models. By adjusting model settings beyond the defaults, we can tailor our models to better fit the specific characteristics of our dataset.

### Purpose of Hyperparameter Tuning:
Hyperparameter tuning is used to optimize model performance by finding the ideal settings such as regularization strengths for Logistic Regression or smoothing parameters for Naive Bayes. This customization helps in maximizing model accuracy and efficiency, ensuring that our models are not just fitting the training data but generalizing well to new, unseen data.

### Tuning Process:

1. **Defining Parameter Grids:**
   - Set up a range of potential values for each model's tunable parameters. For example, possible values of alpha for Naive Bayes and C for Logistic Regression.

2. **Cross-Validation:**
   - Perform cross-validation to test different parameter combinations. This method helps determine which settings improve model performance without fitting too closely to the training set.

3. **Selection of Optimal Parameters:**
   - Choose the parameters that yield the highest validation accuracy, indicating the most effective model configuration.

### Techniques Employed:
- **Grid Search:** An exhaustive method that tests every combination within the parameter grid to identify the best settings.
- **Random Search:** A faster alternative that samples combinations randomly, providing a good balance between thoroughness and efficiency.


---

## Model Evaluation Metrics

To evaluate our machine learning models, we use several key metrics:

1. **Accuracy:**
   - Measures the overall correctness of the model: 
     ```
     Accuracy = (True Positives + True Negatives) / Total Predictions
     ```
### Although intuitive, accuracy might not fully reflect the true performance on imbalanced data.

2. **Precision, Recall, and F1-Score:**
   - **Precision:** Proportion of true positives among predicted positives.
     ```
     Precision = True Positives / (True Positives + False Positives)
     ```
   - **Recall:** Proportion of true positives identified correctly.
     ```
     Recall = True Positives / (True Positives + False Negatives)
     ```
   - **F1-Score:** Harmonic mean of Precision and Recall, useful in imbalanced datasets.
     ```
     F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
     ```

3. **Confusion Matrix:**
   - Visualizes actual versus predicted classifications, highlighting misclassifications.




## Photos


---

## Data Augmentation

To boost model generalization and address limitations due to a small dataset size, we implemented data augmentation techniques. Given the initial dataset consists of only 143 rows with baseline model accuracies of 37% for Naive Bayes and 48% for Logistic Regression, enhancing our dataset became imperative.

### Reasons for Data Augmentation:
**Small Dataset Size:** With only 143 examples, the models lack sufficient data to learn effectively, making overfitting a significant concern. The modest performance of baseline models indicates a need for richer and more varied training data to improve generalization.
- **Overfitting in Hyperparameter Tuning:** During hyperparameter tuning, the model achieved 100% accuracy, indicating severe overfitting to the training data. This further underscored the necessity for a more diverse and robust dataset.
- **Addressing Potential Data Leakage:** While the issue of data leakage is discussed below, ensuring the integrity of augmented data through proper validation splits is a critical part of the augmentation process.

### Techniques Employed:

1. **Synonym Replacement:**
   This technique involves replacing words in the text descriptions with their synonyms from a lexical database like WordNet. 
   - **Impact:** By introducing variations in phrasing without changing the context, synonym replacement helps in developing a model that is robust against different ways of saying the same thing.

2. **Backtranslation:**
   Initially, backtranslation (translating text to a foreign language and then back to the original language) was considered to further diversify the training data.
   - **Outcome:** However, it proved unsuccessful for augmenting more than 70% of the rows effectively due to inconsistencies in the translated content, leading us to rely solely on synonym replacement.

### Implementation of Synonym Replacement:
By selectively altering words in training examples, we created an expanded dataset that better equips the classifier to handle unseen examples. This approach ensures that the model is not just memorizing specific phrases but learning to recognize patterns that are more broadly applicable, significantly enhancing its predictive accuracy.

code photos also and photos of data augemntation text lenth and other till model applied
---

## Evaluating Models on Augmented Data
After augmentation, we train and evaluate models again:

- Compare performance metrics before and after augmentation.
- Look for improvements in accuracy, recall, and precision on augmented test sets.
- Check if the model overcomes previous weaknesses by reducing error rates on particular classes.

---

## Addressing Data Leakage Concerns

Although data augmentation often raises concerns about potential data leakage, the following considerations ensure that no leakage occurred in this project:

1. **Augmentation Applied Only to Training Data:**
   - Synonym replacement and other augmentation techniques were strictly applied to the training set, ensuring the test set remained untouched and independent.

2. **Test Data Integrity Maintained:**
   - Augmented examples did not originate from the test data, and no information from the test set influenced the training process.

3. **Why It May Appear as Leakage:**
   - Augmented data may resemble test data due to linguistic similarities, as generated samples are often semantically related to the original data. However, this resemblance reflects the model’s ability to generalize rather than memorization, as long as augmentation is restricted to the training set.

### References:
- **Deep Learning Book: Training Deep Models (Section 7.5)**  
  [https://www.deeplearningbook.org/contents/optimization.html](https://www.deeplearningbook.org/contents/optimization.html)

- **How to Spot and Avoid Data Leakage in Machine Learning (KDnuggets)**  
  [https://www.kdnuggets.com/2021/08/spot-avoid-data-leakage-machine-learning.html](https://www.kdnuggets.com/2021/08/spot-avoid-data-leakage-machine-learning.html)

- **A Survey of Data Augmentation Techniques for NLP Tasks**  
  [https://arxiv.org/abs/2107.00429](https://arxiv.org/abs/2107.00429)


---

## Neural Network Models

Moving beyond linear models like Naive Bayes and Logistic Regression, we introduce neural networks to better capture complex patterns in the text data.

### Fully Connected Neural Network (NN)
- **Process:**
  - Texts are converted into sequences of integers (tokens).
  - An embedding layer represents words in a dense vector space, capturing semantic similarities.
  - The embeddings are passed through multiple dense layers for feature extraction and classification.

  NN models enable better feature learning compared to fixed representations like TF-IDF vectors.

### RNN / LSTM Model
- **Process:**
  - Employ recurrent layers such as LSTM (Long Short-Term Memory) to capture sequential dependencies in text data.
  - LSTM networks can handle long-term dependencies better than simple feedforward networks, making them ideal for understanding contextual information across longer text inputs.

  LSTM networks excel at capturing the order of words and the context within a sequence, which is crucial for complex disease descriptions.
  ***Better suited for datasets like ours, where text length is variable and longer descriptions contain critical information.***

### Why Neural Networks?
- **Feature Learning:** Neural networks, especially with embeddings, can learn deeper representations of text, making them more effective for classification tasks involving nuanced patterns.
- **Performance Gains:** Traditional models like Logistic Regression and Naive Bayes struggled with accuracy (37% for Naive Bayes and 48% for Logistic Regression), necessitating a shift to more sophisticated architectures.


### Performance Metrics:
| Model       | Original Test Set Accuracy | Expanded Dataset Accuracy |
|-------------|----------------------------|---------------------------|
| NN          | 77.91%                     | 70.05%                    |
| RNN (LSTM)  | 93.02%                     | 73.31%                    |

### Key Observations:
- The RNN model significantly outperformed the NN model on both the original and expanded datasets, reflecting its ability to capture sequential dependencies in text.
- The expanded dataset showed slightly reduced accuracy due to the introduction of more challenging and diverse examples, highlighting the need for further refinement.

### Why Not Other Neural Network Architectures?

1. **Simple Feedforward Networks:**
   - Treat all input features (words) as independent and do not capture sequential or contextual relationships between words.
   - Ineffective for text data with longer descriptions where the order of words significantly impacts meaning. (Reference: "Deep Learning Book" by Ian Goodfellow)

2. **Shallow RNNs or GRUs:**
   - GRUs are computationally lighter than LSTMs but may not perform as well on tasks requiring nuanced understanding of long-term dependencies. 
   - For our dataset, with its complex and often lengthy disease descriptions, LSTMs proved more effective. (Reference: GRU vs LSTM for Sequential Data, Towards Data Science)

---

## Transformers (DistilBERT)
To enhance classification performance, we utilized **DistilBERT**, a lightweight transformer-based language model pre-trained on vast text corpora. DistilBERT builds on the BERT architecture while being smaller and faster, making it efficient for fine-tuning.

### Why DistilBERT?
- **Pre-trained Knowledge:** DistilBERT leverages language patterns and structures learned from extensive pre-training, providing a strong foundation for text classification.
- **Contextual Embeddings:** Unlike traditional embeddings, DistilBERT captures the meaning of words in context, improving accuracy on nuanced text like disease descriptions.
- **Efficiency:** Smaller size and faster training compared to BERT make DistilBERT a practical choice for this project.

### Implementation Overview
1. **Model:** `distilbert-base-uncased` from Hugging Face Transformers was fine-tuned for classification.
2. **Data Preparation:** Texts were tokenized with padding and truncation, ensuring uniform input size.
3. **Training:** Fine-tuned using:
   - Learning Rate: `2e-5`
   - Batch Size: 8
   - Epochs: 3
4. **Evaluation:** Accuracy was measured on the test dataset, showcasing significant improvements.

### Results
- **Transformer Model Accuracy:** Achieved a notable increase in performance compared to traditional and basic neural network models.

### Tools Used
- **Hugging Face Transformers:** For model implementation and fine-tuning.
- **Weights & Biases (W&B):** For tracking experiments and visualizing training metrics.

### What is W&B?
Weights & Biases (W&B) is an AI developer platform that provides tools for training models, fine-tuning models, and leveraging foundation models. It helps in:
- Tracking experiment metrics like loss, accuracy, and other key performance indicators.
- Visualizing model training in real time.
- Organizing and sharing results for collaboration.

### Why Use W&B?
- **Experiment Tracking:** Monitor model performance metrics and compare multiple runs seamlessly.
- **Visualization:** Generate interactive plots for detailed analysis of training and evaluation.
- **Collaboration:** Share project progress and insights with team members easily.

To use W&B, you need to:
1. **Create an Account:** Sign up on the [Weights & Biases website](https://wandb.ai/site/).
2. **API Key:** Generate your personal API key to integrate W&B into your project.
3. **Integration:** Once set up, W&B manages experiment logs and provides dashboards for efficient tracking.

---

## Zero-Shot Classification

To test the model’s capability without direct training on the dataset’s specific classes, we employ a zero-shot classification approach.

### Why Zero-Shot Classification?
Zero-shot classification allows us to classify text into predefined categories without explicitly training the model on labeled examples from those categories. This is especially useful in scenarios where:
- **Limited Training Data:** When labeled data is scarce or unavailable, zero-shot classification provides a practical alternative.
- **Generalization:** It evaluates the model's ability to generalize beyond the data it has been fine-tuned on, showcasing its robustness and adaptability.
- **Domain Adaptation:** By leveraging large pre-trained models, we can apply them to domain-specific tasks (e.g., medical text classification) without additional fine-tuning.

### Implementation
We use the **BART Large MNLI Model**, a transformer-based language model pre-trained for natural language inference tasks. The model:
1. Takes a text input along with a set of candidate labels.
2. Assigns probabilities to each label based on how well the text matches the label.

### Results
| Dataset                              | Accuracy   |
|--------------------------------------|------------|
| Original Dataset                     | **42%** |
| Expanded Dataset (Augmented Dataset) | **39%** |

- **Interpretation:** While the accuracy is lower than fine-tuned models like DistilBERT or LSTM, zero-shot classification demonstrates decent performance considering the model had no prior exposure to the specific disease categories.

### Why Use Zero-Shot Classification?
- **Rapid Prototyping:** It enables quick testing of model capabilities without investing time and resources into data labeling or training.
- **Benchmarking:** Provides a baseline for how well pre-trained models can handle domain-specific tasks.
- **Scalability:** Useful for tasks where new categories might be added dynamically, eliminating the need for retraining.


### References
1. **Hugging Face Documentation:** [Zero-Shot Classification Pipeline](https://huggingface.co/transformers/task_summary.html#zero-shot-classification)  
2. **BART Model Paper:** Lewis, M., Liu, Y., Goyal, N., et al. (2020). [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)  
3. **Practical Use of Zero-Shot Learning:** [Towards Data Science - Zero-Shot Text Classification](https://towardsdatascience.com/how-to-use-zero-shot-classification-for-sentiment-analysis-abf7bd47ad25)
   
By incorporating zero-shot classification, we demonstrate how general-purpose models can handle domain-specific tasks with minimal additional effort, providing insights into their adaptability and potential.

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

