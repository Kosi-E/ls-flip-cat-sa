## importing necessarry libraries
import pandas as pd
import re

import nltk

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


## Reading the dataset
flipkart = pd.read_csv("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/archive (2)/Dataset-SA.csv", encoding='utf-8')
print(flipkart.shape)
print(flipkart.head(100))
 

## Preprocessing the text data
def preprocess_review(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    
    # Tokenize the text
    text = text.split()

    
    # Remove stopwords
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(word, 'v') for word in text if
            word not in set(stopwords.words('english'))]
    return ' '.join(text)


## Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

## vectorize text data(Bag of Words)
CountVec = CountVectorizer(max_features=5000, ngram_range=(1, 3))
X = CountVec.fit_transform(flipkart['Cleaned_Review']).toarray()
y = flipkart['Sentiment'].values

## Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

## Save the model and vectorizer
with open('model.pkl', 'wb') as model_file: 
    pickle.dump(model, model_file)


## GUI Development using Tkinter GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

##Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('CountVec.pkl', 'rb') as vec_file:
    CountVec = pickle.load(vec_file)

## Function to update review and emoji
def update_review(index):
    Cleaned_Review = flipkart['Review'].iloc[index]

    # Preprocess the review
    processed_review = preprocess_review(Cleaned_Review)
    
    # Vectorize the review
    vectorized_review = CountVec.transform([processed_review]).toarray()
    
    # Predict sentiment
    prediction = model.predict(vectorized_review)[0]

    emoji_map = {"positive": happy_emoji, "neutral": neutral_emoji, "negative": angry_emoji}
    emoji_image = emoji_map.get(prediction, neutral_emoji)

    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image
    review_label.config(text=Cleaned_Review)
    
    # Update result label and emoji
    result_label.config(text=f"Sentiment: {prediction}")
    emoji_image = "positive.png" if prediction == "Positive" else "negative.png"
    emoji_img = Image.open(emoji_image)
    emoji_img = emoji_img.resize((50, 50), Image.ANTIALIAS)
    emoji_label.config(image=ImageTk.PhotoImage(emoji_img))
    emoji_label.image = ImageTk.PhotoImage(emoji_img)

    ##Initialize the GUI
root = tk.Tk()
root.title("Flipkart Product Review Sentiment Analysis")

## Load emojis
happy_emoji = ImageTk.PhotoImage(Image.open("happy.png").resize((100, 100), Image.ANTIALIAS))



# Keep valid rows
df_train = df[['Cleaned_Review', 'Sentiment']].dropna()
df_train = df_train[df_train['Cleaned_Review'].str.strip().ne('')]

X_text = df_train['Cleaned_Review'].values
y = df_train['Sentiment'].values

print("Class distribution:\n", pd.Series(y).value_counts())

# 1) Train/Test split (stratified so class balance stays similar)
from sklearn.model_selection import train_test_split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)



