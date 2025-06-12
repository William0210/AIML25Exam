# -----------------------------------------------
# IMPORTING NECESSARY LIBRARIES
# -----------------------------------------------

# NumPy for numerical operations
import numpy as np

# Pandas for handling and analyzing data
import pandas as pd

# Plotting libraries for visual representation
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK for text preprocessing (Natural Language Toolkit)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sklearn libraries for text vectorization, modeling and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ignore warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# LOAD AND DISPLAY DATA FILES
# -----------------------------------------------

# Walk through directory and print all files found in the data folder
import os
for dirname, _, filenames in os.walk('C:/Users/William Olesen/Desktop/AIML25/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# -----------------------------------------------
# READ AND CLEAN THE DATA
# -----------------------------------------------

# Load the CSV file into a DataFrame
# Rename columns for consistency: 'Text' → 'text', 'Label' → 'label'
df = pd.read_csv('C:/Users/William Olesen/Desktop/AIML25/data/sms_spam.csv', encoding='latin1')
df = df.rename(columns={'Text': 'text', 'Label': 'label'})

# Display the first few rows to inspect the structure
print(df.head())

# -----------------------------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------------------------

# This function cleans and tokenizes the text
def preprocess_text(text):
    # Tokenize the input text into individual words
    words = word_tokenize(text)
    
    # Convert all words to lowercase and remove punctuation/symbols
    words = [word.lower() for word in words if word.isalnum()]
    
    # Remove stopwords like "the", "is", "and" etc.
    words = [word for word in words if word not in stopwords.words("english")]
    
    # Return the cleaned text as a single string
    return " ".join(words)

# Apply preprocessing to every message in the dataset
df['text'] = df['text'].apply(preprocess_text)

# -----------------------------------------------
# TEXT VECTORIZATION USING TF-IDF
# -----------------------------------------------

# TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

# Transform the text column into a TF-IDF feature matrix
X = tfidf_vectorizer.fit_transform(df['text']).toarray()

# Target labels (spam/ham)
y = df['label']

# -----------------------------------------------
# SPLIT DATA INTO TRAINING AND TEST SETS
# -----------------------------------------------

# 80% training data, 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------
# TRAINING NAIVE BAYES CLASSIFIER
# -----------------------------------------------

# Initialize the Multinomial Naive Bayes classifier with smoothing parameter alpha
classifier = MultinomialNB(alpha=0.1)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# -----------------------------------------------
# NLTK WRAPPER FOR SKLEARN CLASSIFIER (OPTIONAL)
# -----------------------------------------------

# This class wraps the Scikit-learn classifier to make it compatible with NLTK
from nltk.classify import ClassifierI

class SklearnNLTKClassifier(ClassifierI):
    def __init__(self, classifier):
        self._classifier = classifier
    
    def classify(self, features):
        # Predict a single instance
        return self._classifier.predict([features])[0]
    
    def classify_many(self, featuresets):
        # Predict multiple instances
        return self._classifier.predict(featuresets)
    
    def prob_classify(self, features):
        # Probability not implemented in this wrapper
        raise NotImplementedError("Probability estimation not available.")
    
    def labels(self):
        return self._classifier.classes_

# Instantiate the wrapper
nltk_classifier = SklearnNLTKClassifier(classifier)

# -----------------------------------------------
# MAKE PREDICTIONS ON THE TEST DATA
# -----------------------------------------------

# Use the classifier to predict the test set
y_pred = nltk_classifier.classify_many(X_test)

# -----------------------------------------------
# EVALUATION METRICS AND REPORTS
# -----------------------------------------------

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a full classification report (precision, recall, F1-score)
report = classification_report(y_test, y_pred)

# Format accuracy as a string
acc = f"Accuracy is : {accuracy:.2f}"

# -----------------------------------------------
# DISPLAY TEXTUAL RESULTS IN PLOT
# -----------------------------------------------

plt.figure(figsize=(8, 6), dpi=300)
plt.text(0.5, 0.6, report, fontsize=12, color='darkred', ha='center', va='center',
         bbox=dict(facecolor='white', edgecolor='darkred'))
plt.text(0.5, 0.4, acc, fontsize=12, color='green', ha='center', va='center',
         bbox=dict(facecolor='white', edgecolor='green'))
plt.title('Classification Report (Naive Bayes)')
plt.axis('off')
plt.show()

# -----------------------------------------------
# VISUALIZE THE CONFUSION MATRIX
# -----------------------------------------------

# Create confusion matrix from true and predicted labels
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(4, 3), dpi=200)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
