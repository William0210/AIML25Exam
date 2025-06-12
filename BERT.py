# ----------------------------------------------------
# IMPORTING NECESSARY LIBRARIES
# ----------------------------------------------------

# Core data and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import transformers

# Text processing and NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Torch and Transformers for deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import SGD

# Scikit-learn for data splitting and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Ignore warnings for clean output
import warnings, os
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ----------------------------------------------------
# LOAD AND PREPROCESS THE DATA
# ----------------------------------------------------

# Load the CSV file and rename columns for clarity
df = pd.read_csv('C:/Users/William. O/Desktop/AIML25/data/sms_spam.csv', encoding='latin1')
df = df.rename(columns={'Text': 'text', 'Label': 'label'})

# Define a function to clean the SMS text
def preprocess_text(text):
    words = word_tokenize(text)  # Tokenize the sentence
    words = [w.lower() for w in words if w.isalnum()]  # Keep alphanumeric, lowercase
    words = [w for w in words if w not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

# Apply the preprocessing function to all messages
df['text'] = df['text'].apply(preprocess_text)

# ----------------------------------------------------
# SPLIT DATA AND PREPARE FOR BERT
# ----------------------------------------------------

# Split the cleaned data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Encode labels (spam/ham -> 0/1 or vice versa)
label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
y_train = [label_map[label] for label in y_train]
y_test = [label_map[label] for label in y_test]

# Load the BERT tokenizer and convert texts into input IDs and attention masks
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

# ----------------------------------------------------
# CUSTOM PYTORCH DATASET FOR BERT INPUT
# ----------------------------------------------------

# Define a PyTorch Dataset to work with BERT inputs
class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        # For each data point, return dictionary with input_ids, attention_mask, and label
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# Create Dataset objects
train_dataset = SpamDataset(train_encodings, y_train)
test_dataset = SpamDataset(test_encodings, y_test)

# ----------------------------------------------------
# INITIALIZE BERT MODEL FOR CLASSIFICATION
# ----------------------------------------------------

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_map)  # Number of output labels (spam/ham)
)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----------------------------------------------------
# TRAIN THE MODEL WITH SGD OPTIMIZER
# ----------------------------------------------------

# Set up data loader and optimizer
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = SGD(model.parameters(), lr=5e-5)  # Using SGD instead of AdamW

# Set model to training mode
model.train()

# Run training loop (only 2 epochs for speed)
for epoch in range(2):
    for batch in train_loader:
        # Move batch data to GPU/CPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# ----------------------------------------------------
# EVALUATE THE MODEL ON TEST DATA
# ----------------------------------------------------

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=16)

# Set model to evaluation mode
model.eval()

predictions, true_labels = [], []

# Disable gradient computation during inference
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(batch['labels'].cpu().numpy())

# ----------------------------------------------------
# PERFORMANCE REPORT
# ----------------------------------------------------

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Detailed classification report
print(classification_report(true_labels, predictions))

# ----------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# ----------------------------------------------------

plt.figure(figsize=(4, 3), dpi=200)
sns.heatmap(confusion_matrix(true_labels, predictions), annot=True, fmt='d', cmap='Reds',
            xticklabels=list(label_map.keys()), yticklabels=list(label_map.keys()))
plt.title('BERT Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
