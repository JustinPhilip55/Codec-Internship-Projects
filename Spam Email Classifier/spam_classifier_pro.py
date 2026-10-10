import pandas as pd
import urllib.request
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def download_data():
    """Downloads the standard SMS Spam Collection dataset if not present."""
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    file_name = "sms.tsv"
    if not os.path.exists(file_name):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, file_name)
    return file_name

def main():
    # 1. Load the Data
    print("Loading data...")
    file_path = download_data()
    # The dataset is tab-separated with columns 'label' and 'message'
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    
    # Map labels to binary values (ham = 0, spam = 1)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset loaded successfully. Total emails: {len(df)}")
    print(f"Spam count: {df['label_num'].sum()}, Ham count: {len(df) - df['label_num'].sum()}\n")

    # 2. Split the Data
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 3. Feature Extraction (TF-IDF)
    # Converts text into a matrix of TF-IDF features (better than simple word counts)
    print("Extracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    # 4. Train an Advanced Model (Support Vector Machine)
    print("Training the Support Vector Machine (SVM) model...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_dtm, y_train)

    # 5. Evaluate the Model
    print("Evaluating model...")
    y_pred_class = model.predict(X_test_dtm)
    accuracy = accuracy_score(y_test, y_pred_class)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_class, target_names=['Ham', 'Spam']))

    # 6. Save the Model and Vectorizer (Shows you know deployment basics)
    joblib.dump(model, 'spam_svm_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("\nModel and vectorizer saved as .pkl files for future deployment.")

    # 7. Generate a Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Spam Classifier Confusion Matrix')
    
    # Save the plot as an image instead of just showing it (great for your GitHub README)
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix visual saved as 'confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    main()