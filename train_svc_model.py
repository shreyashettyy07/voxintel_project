import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("final_dataset.xlsx")

df["english"] = df["english"].str.lower().str.strip()
df["merged_intent"] = df["merged_intent"].str.lower().str.strip()

X = df["english"]
y = df["merged_intent"]
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=7000
)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LinearSVC(C=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred, zero_division=0))