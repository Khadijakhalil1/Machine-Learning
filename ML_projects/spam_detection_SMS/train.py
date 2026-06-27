import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Dataset load karo
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Label encode karo
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text transform karo
df['transformed'] = df['text'].apply(transform_text)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df['transformed'], df['label'], test_size=0.2, random_state=42
)

# Vectorizer fit karo
tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)

# Model train karo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save karo
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("Model aur Vectorizer successfully save ho gaye!")