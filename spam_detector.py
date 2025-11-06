from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

df = pd.read_csv('sms_spam.csv', sep='\t', names=['label', 'message'])

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

y_pred = model.predict(X_test_counts)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

your_msg = ["Congrats! You won a free prize. Click now!"]
msg_counts = vectorizer.transform(your_msg)
prediction = model.predict(msg_counts)
print('Prediction:', prediction[0])  # Will show if spam or ham (not spam)
