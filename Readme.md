
# Sports vs. Politics Text Classifier 

## Project Overview
This project implements a binary text classification system to distinguish between **Sports** and **Politics** news articles. 

Using the **AG News** dataset, we constructed a balanced dataset of 25,000 documents and performed a comparative analysis of three feature representation techniques (Bag of Words, TF-IDF, N-Grams) across three supervised learning algorithms:
- **Multinomial Naive Bayes (MNB)**
- **Logistic Regression (LR)**
- **Linear Support Vector Machine (SVM)**

The best-performing model achieved **~95.8% accuracy** using **Linear SVM with TF-IDF features**.

## Dataset
- **Source:** [AG News Dataset](https://huggingface.co/datasets/ag_news)
- **Classes:** - `Politics` (derived from "World" class)
  - `Sports` (derived from "Sports" class)
- **Size:** 25,000 balanced samples (12,500 per class)
- **Split:** 80% Training, 20% Testing

## Results Summary

| Feature Representation | Model | Accuracy |
| :--- | :--- | :--- |
| **Bag of Words** | Naive Bayes | 91.50% |
| | Logistic Regression | 93.20% |
| | Linear SVM | 92.80% |
| **TF-IDF** | Naive Bayes | 92.10% |
| | Logistic Regression | 94.50% |
| | **Linear SVM** | **95.80%** 🏆 |
| **N-Grams (1,2)** | Naive Bayes | 93.40% |
| | Logistic Regression | 95.10% |
| | Linear SVM | 95.75% |

## 🚀 How to Run the Code

### 1. Prerequisites
Ensure you have Python installed (3.8+ recommended). You will need the following libraries:
```bash
pip install pandas numpy scikit-learn datasets

```

### 2. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/sports-politics-classifier.git
cd sports-politics-classifier

```

### 3. Run the Script

Execute the main Python script to load data, train models, and see the results:

```bash
python B23CM1062_prob4.py

```

### 4. Expected Output

The script will:

1. Download/Load the AG News dataset.
2. Filter and balance the data.
3. Train 9 different model variations (3 features x 3 classifiers).
4. Print the accuracy table.
5. Run a live demo prediction on sample text.

---

## Author

**Varchasva Raj Saxena**
 *B23CM1062* 
 Artificial Intelligence & Data Sciences

Indian Institute of Technology, Jodhpur

```

