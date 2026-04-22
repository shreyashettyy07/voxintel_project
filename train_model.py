import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_excel("cleaned_dataset.xlsx")

X = df["english"]
y = df["merged_intent"]


vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)



model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

test_query = ["What is the weather today?"]


test_vec = vectorizer.transform(test_query)


prediction = model.predict(test_vec)

print("\nPredicted Intent:", prediction[0])