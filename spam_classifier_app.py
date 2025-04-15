import streamlit as st
import pandas as pd
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", header=None)
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].str.lower().apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

# Vectorize
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("📧 Spam Classifier with Logistic Regression")
st.write("Enter a message and see if it's classified as **Spam** or **Ham**.")

user_input = st.text_area("Enter your message:", "")

if st.button("Classify"):
    # Preprocess input
    cleaned_input = ''.join([char for char in user_input.lower() if char not in string.punctuation])
    vectorized_input = tfidf.transform([cleaned_input])
    
    prediction = model.predict(vectorized_input)[0]
    prediction_proba = model.predict_proba(vectorized_input)[0][1]

    if prediction == 1:
        st.error(f"🚨 Spam ({prediction_proba:.2f} probability)")
    else:
        st.success(f"✅ Ham ({1 - prediction_proba:.2f} probability)")
