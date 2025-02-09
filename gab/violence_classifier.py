import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (Replace with actual data)
# Assuming df has columns: ['Text', 'CV']
# Text = raw text data, CV = binary label (0 or 1)
df = pd.read_table('GabHateCorpus_annotations.tsv')
df = df[['ID','Annotator', 'Text','CV']]

# Step 1: Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['CV'], test_size=0.2, random_state=42)

# Step 2: Convert Text to TF-IDF Features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features for performance
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 3: Train Logistic Regression Model
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

# Step 4: Predictions & Evaluation
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
print(accuracy)
print(report)

