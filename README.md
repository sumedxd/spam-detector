
Spam SMS Detector
Overview
This is a simple machine learning project that detects spam SMS messages using Python. It trains a model on a public dataset and predicts whether a given SMS is spam or not.

Features
Loads and preprocesses SMS text

Trains a spam classifier model (Naive Bayes)

Evaluates accuracy of the model

Predicts spam/ham for new messages

Dataset
Uses the UCI SMS Spam Collection Dataset.

How to Use
Install Requirements

bash
pip install pandas scikit-learn numpy nltk
Download Dataset

Download SMSSpamCollection from the UCI site.

Place it in the project folder as sms_spam.csv.

Run the Script

Edit and run spam_detector.py:

bash
python spam_detector.py
Test on Your Own SMS Text

Add your own message in spam_detector.py to check if it is spam.

Files
spam_detector.py: Main Python script

sms_spam.csv: Raw SMS dataset

requirements.txt: Python dependencies

Example
text
Input message: "Congrats! You won a free prize. Click now!"
Prediction: spam

