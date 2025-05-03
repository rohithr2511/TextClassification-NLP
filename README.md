# 📝 NLP Text Classification — Assignment 1

This project focuses on developing a **text classification pipeline** using basic NLP techniques and machine learning algorithms. The goal is to correctly classify text samples into their respective categories using a **Logistic Regression** classifier trained on **TF-IDF features**.

---

## 🎯 Objective

* Preprocess raw text data
* Transform text into numerical features using TF-IDF
* Train a machine learning classifier (Logistic Regression)
* Evaluate performance using accuracy and confusion matrix
* Analyze misclassifications for insight

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── text_classification_assignment.ipynb   # Main Jupyter Notebook
├── data/
│   └── text_class.csv                         # Input dataset
├── outputs/
│   ├── confusion_matrix.png
│   └── accuracy_score.txt
├── README.md
└── requirements.txt
```

---

## 📊 Task 1: Data Exploration (2 Marks)

* Loaded dataset using `pandas`.
* Previewed first 5 rows.
* Identified target labels and class distribution.

### ✅ **Insights:**

* Class distribution was slightly imbalanced — some categories had fewer examples.
* No missing values in the dataset.
* Some text samples contained special characters and URLs that needed cleaning.

---

## 🧹 Task 2: Text Preprocessing (3 Marks)

* Converted text to lowercase.
* Removed punctuation, special characters, and digits.
* Tokenized text and removed stopwords using `nltk`.

### ✅ **Insights:**

* Stopword removal helped reduce noise and improved model generalization.
* Punctuation removal prevented misinterpretation of special tokens.
* Preprocessing reduced vocabulary size significantly (by 40–60%).

---

## 🏗️ Task 3: Train a Classifier (3 Marks)

* Used `TfidfVectorizer` to convert cleaned text into numerical features.
* Split data: **80% for training**, **20% for testing**.
* Trained `LogisticRegression` using `sklearn`.

### ✅ **Insights:**

* Logistic Regression performed well on sparse high-dimensional TF-IDF features.
* Training time was short; good for fast iterations.
* TF-IDF captured relevant word frequency patterns, aiding classification.

---

## 📈 Task 4: Evaluate the Model (2 Marks)

* Achieved **Test Accuracy: \~84%**
* Generated:

  * Confusion matrix
  * Classification report
  * Accuracy score

### ✅ **Insights:**

* Certain classes were confused due to similar vocabulary (e.g., tech vs. business).
* Precision/recall imbalance highlighted areas for improvement.
* Adding lemmatization or bigrams could further improve accuracy.

---

## 🧠 Final Takeaways

* **Preprocessing** is crucial for cleaning real-world text data.
* **TF-IDF + Logistic Regression** is a strong baseline for text classification.
* Analyzing **confusion matrix** gives clues about label-specific weaknesses.
* Model could be further improved using:

  * Bigrams/trigrams in TF-IDF
  * Lemmatization or stemming
  * Advanced models like Naive Bayes, SVMs, or fine-tuned BERT

---

## ⚙️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/nlp-text-classification.git
   cd nlp-text-classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook notebooks/text_classification_assignment.ipynb
   ```

---

## 📦 Requirements

* `pandas`
* `scikit-learn`
* `nltk`
* `matplotlib`
* `seaborn`

Download NLTK stopwords before running:

```python
import nltk
nltk.download('stopwords')
```

