# -----------------------------------------------
# IMPORTING NECESSARY LIBRARIES
# -----------------------------------------------

# Core libraries for data handling and computation
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Natural Language Toolkit (NLTK) for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Scikit-learn modules for ML and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ignore warning messages for clean output
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# LOAD AND INSPECT THE DATA
# -----------------------------------------------

# Print all files found in the given directory to ensure data is available
import os
for dirname, _, filenames in os.walk('C:/Users/William. O/Desktop/AIML25/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the CSV file and rename relevant columns for clarity
df = pd.read_csv('C:/Users/William. O/Desktop/AIML25/data/sms_spam.csv', encoding='latin1')
df = df.rename(columns={'Text': 'text', 'Label': 'label'})

# Show first few rows of the dataset
print(df.head())

# -----------------------------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------------------------
nltk.download('punkt_tab')
nltk.download('stopwords')
# Define a function to clean and preprocess the raw SMS messages
def preprocess_text(text):
    # Tokenize the message into individual words
    words = word_tokenize(text)
    
    # Convert to lowercase and remove non-alphanumeric characters
    words = [word.lower() for word in words if word.isalnum()]
    
    # Remove common stopwords (e.g., "and", "is", "the")
    words = [word for word in words if word not in stopwords.words("english")]
    
    # Recombine words into a cleaned sentence
    return " ".join(words)

# Apply the cleaning function to each message in the dataset
df['text'] = df['text'].apply(preprocess_text)

# -----------------------------------------------
# FEATURE EXTRACTION WITH TF-IDF
# -----------------------------------------------

# Convert text data into TF-IDF vectors (weighted word frequencies)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['text']).toarray()

# Define the target variable (spam or ham)
y = df['label']

# -----------------------------------------------
# SPLIT DATA INTO TRAIN AND TEST SETS
# -----------------------------------------------

# Reserve 20% of the data for testing and set a random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------
# TRAINING LOGISTIC REGRESSION CLASSIFIER
# -----------------------------------------------

# Initialize the Logistic Regression model
# max_iter increased to ensure convergence during training
logistic_classifier = LogisticRegression(max_iter=1000)

# Fit the model on the training data
logistic_classifier.fit(X_train, y_train)

# -----------------------------------------------
# OPTIONAL: WRAP SKLEARN MODEL FOR NLTK COMPATIBILITY
# -----------------------------------------------

# NLTK wrapper for Scikit-learn classifier, not strictly required, but used for consistency
from nltk.classify import ClassifierI

class SklearnNLTKClassifier(ClassifierI):
    def __init__(self, classifier):
        self._classifier = classifier
    
    def classify(self, features):
        return self._classifier.predict([features])[0]
    
    def classify_many(self, featuresets):
        return self._classifier.predict(featuresets)
    
    def prob_classify(self, features):
        raise NotImplementedError("Probability estimation not available.")
    
    def labels(self):
        return self._classifier.classes_

# Wrap the classifier for compatibility
nltk_classifier = SklearnNLTKClassifier(logistic_classifier)

# -----------------------------------------------
# MAKE PREDICTIONS AND EVALUATE PERFORMANCE
# -----------------------------------------------

# Use the trained model to predict the test set
y_pred = nltk_classifier.classify_many(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report with precision, recall, F1-score
report = classification_report(y_test, y_pred)

# Format accuracy for display
acc = f"Accuracy is : {accuracy:.2f}"

# -----------------------------------------------
# VISUALIZATION: CLASSIFICATION REPORT
# -----------------------------------------------

plt.figure(figsize=(8, 6), dpi=300)
plt.text(0.5, 0.6, report, fontsize=12, color='darkred', ha='center', va='center',
         bbox=dict(facecolor='white', edgecolor='darkred'))
plt.text(0.5, 0.4, acc, fontsize=12, color='green', ha='center', va='center',
         bbox=dict(facecolor='white', edgecolor='green'))
plt.title('Classification Report (Logistic Regression)')
plt.axis('off')
plt.show()

# -----------------------------------------------
# VISUALIZATION: CONFUSION MATRIX
# -----------------------------------------------

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot as heatmap
plt.figure(figsize=(4, 3), dpi=200)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
