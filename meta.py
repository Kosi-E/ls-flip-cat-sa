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

plt.style.use('ggplot')

## Reading the dataset
df = pd.read_csv("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/archive (2)/Dataset-SA.csv")

print(df.size())
