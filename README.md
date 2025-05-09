# 📰 Exposing the Truth: Fake News Detection Using NLP

## 🎯 Project Statement

The widespread dissemination of fake news across social media and news platforms poses serious threats to public trust, democracy, and societal stability. This project addresses the binary classification problem of distinguishing fake news from real news using Natural Language Processing (NLP). By leveraging machine learning algorithms on textual data, we aim to create an automated system that flags misinformation before it spreads.

---

## 📌 Objectives

- Apply NLP techniques to clean, tokenize, and vectorize article content.
- Compare performance of Logistic Regression and Random Forest models.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Optimize the pipeline for real-world, real-time detection scenarios.

---

## 📄 Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Type**: Textual (unstructured)
- **Records**: ~44,000 articles
- **Columns**:
  - `title`: Headline of the article
  - `text`: Full article content
  - `subject`: Category of the article
  - `date`: Published date
  - `label`: `FAKE` or `REAL` (Target variable)

---

## 🧹 Data Preprocessing

- Removal of missing and duplicate records
- Text cleaning: lowercasing, punctuation removal, and number filtering
- Tokenization, stopword removal, and lemmatization using NLTK
- Label encoding (`FAKE` → 0, `REAL` → 1)
- Text vectorization using TF-IDF (Top 5000 features)

---

## 📊 Exploratory Data Analysis

- **Univariate**: Word clouds, frequency plots, article length distributions
- **Bivariate**: Correlation between article length and label, subject vs. label
- **Insights**:
  - Fake news often uses sensational words like “breaking” and “shocking”
  - Real articles are generally longer and more information-rich

---

## 🏗️ Feature Engineering

- Extracted article length as a feature
- Extracted day and month from date field
- N-gram generation (unigrams to trigrams)
- Dimensionality reduction using top-k TF-IDF scores

---

## 🤖 Model Building

- **Models Used**:
  - Logistic Regression (baseline, interpretable)
  - Random Forest (handles complexity and feature interactions)

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC and ROC Curve Visualization

---

## 📈 Visualization of Results

- Confusion Matrix heatmaps
- ROC Curve comparisons for both models
- Feature importance from Random Forest
- Top 20 significant words from Logistic Regression coefficients

---

## 🛠 Tools & Technologies

| Category         | Tools/Packages                                   |
|------------------|--------------------------------------------------|
| Programming      | Python                                            |
| Libraries        | pandas, numpy, sklearn, nltk, seaborn, matplotlib, wordcloud |
| NLP              | NLTK, Scikit-learn’s TF-IDF Vectorizer           |
| IDE              | Jupyter Notebook, Google Colab                   |
| Visualization    | matplotlib, seaborn, wordcloud                   |

---

## 📁 File Structure

├── Dataset.csv # Labeled news articles
├── source_code.py # Python code for training and evaluation
└── read.md # This documentation

## Team Members and Roles
| Name         | Responsibility                |
| ------------ | ----------------------------- |
| Sornamalya T | Data cleaning & preprocessing |
| Thasmiya M   | EDA & Feature engineering     |
| Deepika D    | Model development & tuning    |
| Abirami B    | Visualization & documentation |


