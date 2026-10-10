# Codec Technologies Internship Projects

This repository contains the completed projects for the Machine Learning and Artificial Intelligence internship at Codec Technologies.

## Project 1: Spam Email/SMS Classifier
A robust Natural Language Processing (NLP) pipeline built to classify text messages as either 'Spam' or 'Ham' (legitimate). 

### 🛠️ Technologies Used
* **Language:** Python
* **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib
* **Techniques:** TF-IDF Vectorization, Support Vector Machines (SVM)

### 📊 Model Performance
The model was trained on the standard SMS Spam Collection dataset and achieved an **Accuracy of 98.78%**. 
*The confusion matrix (`confusion_matrix.png`) and the exported deployment models (`.pkl` files) are included in the project directory.*

---

## Project 2: Speech-to-Text Transcription
An automated Python tool that converts spoken audio into written text using the Google Web Speech API.

### 🛠️ Technologies Used
* **Language:** Python
* **Libraries:** SpeechRecognition, urllib, os
* **Techniques:** API Integration, Ambient Noise Calibration, Automated Audio Data Processing

### ⚙️ Features
* **Automated Asset Handling:** The script automatically fetches a standard acoustic testing `.wav` file (Harvard Sentences).
* **Noise Calibration:** Adjusts processing based on the ambient noise of the specific audio file before transcription.
* **Automated Output:** Generates a formatted `transcription_output.txt` file containing the recognized speech. 

---
### 🚀 How to Run the Projects
Each project is contained in its respective folder. 
1. Navigate to the desired project folder.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment and install dependencies: `pip install -r requirements.txt`
4. Run the respective python script (`spam_classifier_pro.py` or `speech_transcriber_pro.py`).