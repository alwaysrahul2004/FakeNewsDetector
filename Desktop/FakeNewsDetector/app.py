import streamlit as st
import joblib
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# Sidebar
st.sidebar.title("📰 Fake News Detector Info")
st.sidebar.markdown("""
This app uses a **Logistic Regression** model trained on real & fake news to detect misinformation.
- Input any news article.
- Get prediction + model confidence.
- Visuals like word cloud & keyword highlights.
""")

# Streamlit UI
st.title("📰 Fake News Detector")
user_input = st.text_area("Enter the news content here:")

if st.button("Predict"):
    cleaned_text = clean(user_input)
    vectorized_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_input)[0]
    probas = model.predict_proba(vectorized_input)[0]

    # Show result
    if prediction == 0:
        st.error(f"⚠️ This news is **FAKE!** (Confidence: {probas[0]*100:.2f}%)")
    else:
        st.success(f"✅ This news is **REAL!** (Confidence: {probas[1]*100:.2f}%)")

    # Word cloud
    st.subheader("📊 Word Cloud of Input")
    wordcloud = WordCloud(width=800, height=300, background_color='black').generate(cleaned_text)
    plt.figure(figsize=(10, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Highlight top words
    st.subheader("🔍 Top Influential Words")
    vec = vectorizer.transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    top_indices = vec.toarray()[0].argsort()[::-1][:5]
    top_words = [feature_names[i] for i in top_indices if vec.toarray()[0][i] > 0]

    if top_words:
        st.write("These words had the biggest impact on the prediction:")
        st.code(", ".join(top_words))
    else:
        st.write("No strong keywords found.")
