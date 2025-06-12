# Define imports and extract libraries
import os
import requests
import pandas as pd
from zipfile import ZipFile
from io import BytesIO

# Define folder/file paths
data_dir = "data"
csv_path = os.path.join(data_dir, "sms_spam.csv")
dataset_txt = os.path.join(data_dir, "SMSSpamCollection")

# Check if CSV already exists
if os.path.exists(csv_path):
    print(f"sms.spam.csv already exists.")
else:
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract dataset from UC Irvine archive
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(data_dir)

    # Convert to CSV
    df = pd.read_csv(dataset_txt, sep='\t', header=None, names=['Label', 'Text'])
    df.to_csv(csv_path, index=False)
    print(f"Downloaded and saved CSV at: {csv_path}")

    # Delete README
    for fname in os.listdir(data_dir):
        if "readme" in fname.lower() or fname == "SMSSpamCollection":
            os.remove(os.path.join(data_dir, fname))
            print(f"Deleted file: {fname}")
