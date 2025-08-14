## importing necessarry libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

dd = plt.style.use('ggplot')

## Reading the dataset
df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/archive (2)/Dataset-SA.csv")
print(df.shape)
print(df.head(100))

## EDA
ax = df['Rate'].value_counts().sort_index().plot(kind='bar',title='Count Of Reviews by Rates', figsize=(10, 5), color=['#FF5733'])
ax.set_xlabel('Rate')
ax.set_ylabel('Count of Reviews')
plt.show()


## Preprocessing the text data
def preprocess_review(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)