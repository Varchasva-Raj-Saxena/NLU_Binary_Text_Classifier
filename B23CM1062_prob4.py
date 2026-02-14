import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("--- Loading AG News Dataset (Download happens once) ---")

dataset = load_dataset("ag_news")

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
full_df = pd.concat([train_df, test_df])

print(f"Original Dataset Size: {len(full_df)} documents")

filtered_df = full_df[full_df['label'].isin([0, 1])].copy()

filtered_df['label_name'] = filtered_df['label'].map({0: 'Politics', 1: 'Sports'})

df_politics = filtered_df[filtered_df['label'] == 0].sample(12500, random_state=42)
df_sports = filtered_df[filtered_df['label'] == 1].sample(12500, random_state=42)
final_df = pd.concat([df_politics, df_sports])

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final Dataset for Training: {len(final_df)} documents")

print("\n--- Splitting Data (80% Train, 20% Test) ---")

X_train, X_test, y_train, y_test = train_test_split(
    final_df['text'],
    final_df['label'],
    test_size=0.2,
    random_state=42,
    stratify=final_df['label']
)

features = {
    "Bag of Words": CountVectorizer(stop_words='english', max_features=10000),
    "TF-IDF": TfidfVectorizer(stop_words='english', max_features=10000),
    "N-Grams (1,2)": TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
}

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear)": LinearSVC(dual=False)
}

print(f"\n{'FEATURE':<20} | {'MODEL':<20} | {'ACCURACY':<10}")
print("-" * 55)

for feat_name, vectorizer in features.items():
    for model_name, clf in models.items():
        pipe = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', clf)
        ])

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"{feat_name:<20} | {model_name:<20} | {acc:.4f}")

print("\n--- Live Demo (Using N-Grams + SVM) ---")

demo_pipe = Pipeline([
    ('ngram', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)),
    ('svm', LinearSVC(dual=False))
])

demo_pipe.fit(X_train, y_train)

samples = [
    "The prime minister announced a new tax policy for the upcoming election.",
    "Ronaldo scored a magnificent goal in the final minutes of the match.",
    "The treaty was signed by three nations in Geneva yesterday."
]

preds = demo_pipe.predict(samples)

for text, p in zip(samples, preds):
    label = "Sports" if p == 1 else "Politics"
    print(f"Input: ...{text[-30:]} -> Prediction: {label}")
