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
print("STEP 1: CSV loaded → shape:", flipkart.shape, "| columns:", list(flipkart.columns))
 

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

## Apply preprocessing to create the Cleaned_Review column
flipkart['Cleaned_Review'] = flipkart['Review'].astype(str).apply(preprocess_review)
print(flipkart[['Review', 'Cleaned_Review']].head(10))
print("STEP 2: Creating Cleaned_Review ...")

flipkart['Sentiment'] = flipkart['Sentiment'].astype(str).str.strip().str.lower()
print("Sentiment column after cleaning:", flipkart['Sentiment'].unique())

df_train = flipkart[['Cleaned_Review', 'Sentiment']].dropna()
df_train = df_train[df_train['Cleaned_Review'].str.strip() != '']

X_text = df_train['Cleaned_Review'].astype(str).values
y = df_train['Sentiment'].astype(str).str.strip().str.lower().values

print("STEP 3: Class distribution:\n", pd.Series(y).value_counts())

from sklearn.model_selection import train_test_split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.feature_extraction.text import CountVectorizer
CountVec = CountVectorizer(max_features=5000, ngram_range=(1, 3))

X_train = CountVec.fit_transform(X_train_text)
X_test  = CountVec.transform(X_test_text)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report
print("STEP 4: \nClassification report:\n", classification_report(y_test, model.predict(X_test), digits=3))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on the test split you already created
y_pred = model.predict(X_test)

# Ensure consistent class order
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("STEP 4b: Confusion matrix (rows=true, cols=pred):\n", cm)

# Plot and save
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax, values_format='d', colorbar=False)  # no style/colors set
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

cm_norm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
fig, ax = plt.subplots(figsize=(5,5))
disp_norm.plot(ax=ax, values_format=".2f", colorbar=False)
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=200)
plt.show()

## Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

## vectorize text data(Bag of Words)
print("STEP 5: Vectorizing with CountVectorizer(max_features=5000, ngram_range=(1,3)) ...")
CountVec = CountVectorizer(max_features=5000, ngram_range=(1, 3))
X = CountVec.fit_transform(flipkart['Cleaned_Review'])
y = flipkart['Sentiment'].values
print("Vectorized. X.shape =", X.shape, "| vocab_size =", len(CountVec.vocabulary_))

## Train Naive Bayes model
print("STEP 6: Training MultinomialNB ...")
model = MultinomialNB()
model.fit(X, y)
print("Model trained. Classes:", model.classes_)

## Save the model and vectorizer
with open('model.pkl', 'wb') as model_file: 
    pickle.dump(model, model_file)
with open('CountVec.pkl', 'wb') as vec_file:
    pickle.dump(CountVec, vec_file)
import os
print("STEP 7: Saved model →", os.path.abspath('model.pkl'))
print("STEP 7: Saved vectorizer →", os.path.abspath('CountVec.pkl'))

## GUI Development using Tkinter GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

## Function to update review and emoji
def update_review(index):
    Cleaned_Review = flipkart['Review'].iloc[index]

    # Preprocess the review
    processed_review = preprocess_review(Cleaned_Review)

    
    # Vectorize the review
    vectorized_review = CountVec.transform([processed_review])
    
    # Predict sentiment
    prediction = model.predict(vectorized_review)[0]
    prediction = str(prediction).strip().lower()

    emoji_map = {"positive": happy_emoji, "neutral": neutral_emoji, "negative": angry_emoji}
    emoji_image = emoji_map.get(prediction, neutral_emoji)

    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image
    review_label.config(text=Cleaned_Review)
    
    # Update result label and emoji
    result_label.config(text=f"Sentiment: {prediction.capitalize()}")

    ##Initialize the GUI
root = tk.Tk()
root.title("Flipkart Product Review Sentiment Analysis")

## Load emojis
happy_emoji = ImageTk.PhotoImage(Image.open("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/ls-flip-cat-sa/emoji_image/happy_emoji.jpg").resize((100, 100)))
neutral_emoji = ImageTk.PhotoImage(Image.open("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/ls-flip-cat-sa/emoji_image/neutral_emoji.png").resize((100, 100)))
angry_emoji = ImageTk.PhotoImage(Image.open("C:/Users/Asus/OneDrive/Documents/Learning Saint/Task 1/ls-flip-cat-sa/emoji_image/angry_emoji.jpg").resize((100, 100)))


# Setup the labels and buttons
emoji_label = tk.Label(root)
emoji_label.grid(row=0, column=0, padx=10, pady=10)

review_label = tk.Label(root, wraplength=300, justify="center")
review_label.grid(row=1, column=0, padx=10, pady=10)

# NEW: result label for showing sentiment
result_label = tk.Label(root, text="Sentiment: ", font=("Arial", 12))
result_label.grid(row=2, column=0, padx=10, pady=10)

current_index = [0]

def next_review():
    current_index[0] = (current_index[0] + 1) % len(flipkart)
    update_review(current_index[0])

next_button = ttk.Button(root, text="Next Review", command=next_review)
next_button.grid(row=4, column=0, padx=10, pady=10)

##
from tkinter import filedialog, messagebox
import time, glob
print("STEP 8: Creating stats label")
stats_label = tk.Label(root, text="", font=("Arial", 10))
stats_label.grid(row=5, column=0, padx=10, pady=6)

def update_stats():
    print("STEP 8.1: Updating stats")
    counts = flipkart['Sentiment'].astype(str).str.lower().value_counts()
    pos = counts.get('positive', 0); neu = counts.get('neutral', 0); neg = counts.get('negative', 0)
    stats_label.config(text=f"Counts → Positive: {pos}  |  Neutral: {neu}  |  Negative: {neg}")
    print(f"STEP 8.2: Stats → pos={pos}, neu={neu}, neg={neg}")

def load_new_csv():
    print("STEP 9: Choose CSV…")
    path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    if not path:
        print("STEP 9.x: No file selected"); return

    t0 = time.time()
    print("STEP 9.1: Reading CSV →", path)
    try:
        df_new = pd.read_csv(path, encoding='utf-8')
    except Exception as e:
        print("STEP 9 ERR:", e); messagebox.showerror("Load Error", str(e)); return

    if 'Review' not in df_new.columns:
        messagebox.showerror("Error", "CSV must contain a 'Review' column."); return

    print("STEP 9.2: Preprocessing new reviews")
    df_new['Cleaned_Review'] = df_new['Review'].astype(str).apply(preprocess_review)

    print("STEP 9.3: Predicting sentiments")
    X_new = CountVec.transform(df_new['Cleaned_Review'])
    df_new['Sentiment'] = model.predict(X_new)

    print("STEP 9.4: Merging")
    global flipkart
    keep = ['Review', 'Sentiment', 'Cleaned_Review']
    flipkart = pd.concat([flipkart[keep], df_new[keep]], ignore_index=True)

    print(f"STEP 9 done: {len(df_new)} rows in {time.time()-t0:.2f}s")
    update_stats(); alert_if_negative_spike()
    messagebox.showinfo("Done", f"Processed {len(df_new)} new reviews.")

    ## Robot helper: load a CSV directly from a file path (no dialog)
def load_new_csv_from_path(path):
    print("ROBOT: Loading", path)
    try:
        df_new = pd.read_csv(path, encoding='utf-8')
    except Exception as e:
        print("ROBOT ERR: read failed →", e); return

    if 'Review' not in df_new.columns:
        print("ROBOT: skipped (no 'Review' column)"); return

    # preprocess + predict
    df_new['Cleaned_Review'] = df_new['Review'].astype(str).apply(preprocess_review)
    X_new = CountVec.transform(df_new['Cleaned_Review'])
    df_new['Sentiment'] = model.predict(X_new)

    # merge into main df
    global flipkart
    keep = ['Review', 'Sentiment', 'Cleaned_Review']
    flipkart = pd.concat([flipkart[keep], df_new[keep]], ignore_index=True)

    # optional: dedupe if same file seen twice
    flipkart.drop_duplicates(subset=['Review'], keep='last', inplace=True)

    update_stats()
    alert_if_negative_spike()   # uses default 30%
    print(f"ROBOT: processed {len(df_new)} rows from {os.path.basename(path)}")


def alert_if_negative_spike(threshold=0.30):
    frac_neg = (flipkart['Sentiment'].astype(str).str.lower() == 'negative').mean()
    print(f"STEP 10: Negative share → {frac_neg:.2%} (threshold {threshold:.0%})")
    if frac_neg >= threshold: 
        messagebox.showwarning("Alert", f"High negative share detected: {frac_neg:.0%}")

def retrain_model():
    print("STEP 11: Retraining on ALL current data")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    cv_new = CountVectorizer(max_features=5000, ngram_range=(1, 3))
    X_all = cv_new.fit_transform(flipkart['Cleaned_Review'].astype(str))
    y_all = flipkart['Sentiment'].astype(str).str.lower()
    mdl_new = MultinomialNB().fit(X_all, y_all)

    # swap in memory + save
    global CountVec, model
    CountVec, model = cv_new, mdl_new
    with open('CountVec.pkl','wb') as f: pickle.dump(CountVec, f)
    with open('model.pkl','wb') as f: pickle.dump(model, f)
    print("STEP 11.1: Saved →", os.path.abspath('model.pkl'), "and", os.path.abspath('CountVec.pkl'))
    messagebox.showinfo("Retrained", "Model + vectorizer updated.")

print("STEP 12: Creating buttons (Load CSV, Retrain)")
load_button    = ttk.Button(root, text="Load New Reviews (CSV)", command=load_new_csv)
load_button.grid(row=6, column=0, padx=10, pady=6)

retrain_button = ttk.Button(root, text="Retrain Model", command=retrain_model)
retrain_button.grid(row=7, column=0, padx=10, pady=6)

## Robot watcher: auto-scan a folder every 30s for new CSVs
watch_folder  = r"C:\Users\Asus\Downloads"
scan_every_ms = 30_000

robot_enabled   = [False]
processed_files = set()

def robot_tick():
    if not robot_enabled[0]:
        return
    for path in glob.glob(os.path.join(watch_folder, "*.csv")):
        if path not in processed_files:
            load_new_csv_from_path(path)
            processed_files.add(path)
    root.after(scan_every_ms, robot_tick)

def toggle_robot():
    robot_enabled[0] = not robot_enabled[0]
    robot_button.config(text=f"Robot: {'ON' if robot_enabled[0] else 'OFF'}")
    if robot_enabled[0]:
        print("ROBOT: watching", watch_folder)
        robot_tick()
    else:
        print("ROBOT: paused")

robot_button = ttk.Button(root, text="Robot: OFF", command=toggle_robot)
robot_button.grid(row=8, column=0, padx=10, pady=6)
print("STEP 12.1: Robot watcher initialized →", watch_folder)

print("STEP 13: Initializing stats")
update_stats()


print("STEP 14: GUI ready → showing first review …")
# Initialize with the first review
update_review(current_index[0])
root.mainloop()



