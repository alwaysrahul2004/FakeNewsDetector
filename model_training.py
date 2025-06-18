import pandas as pd
import re
import nltk
import os
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')
fake['label'] = 0
real['label'] = 1
data = pd.concat([fake, real])[['text', 'label']].sample(frac=1).reset_index(drop=True)

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

data['text'] = data['text'].apply(clean)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy}")

# ✅ Ensure 'models' folder exists
os.makedirs('models', exist_ok=True)

# Save model and vectorizer
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
