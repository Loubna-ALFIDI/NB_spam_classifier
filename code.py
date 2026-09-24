import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# step 1: load data 
df = pd.read_csv('emails.csv')
print(df.head())

# step 2: exploring data
rows, cols = df.shape
print(f"{rows} rows, {cols} columns")

# number of rows and columns
print(df['spam'].value_counts())
# duplicates
print(df.duplicated().sum())
# missing values
print(df.isna().sum())

# inspecting the duplicates
print(df[df.duplicated(keep=False)])

# step 3: cleaning data
df = df.drop_duplicates()
rows, cols = df.shape
print(f"{rows} rows, {cols} columns")

# step 4: splitting data
x, y = df['text'], df['spam']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y
)

#step 5: extraxting features
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

print(x_train_tfidf.shape)

#step 6: training the model
model = MultinomialNB()
model.fit(x_train_tfidf, y_train)

#step 7: test
y_pred = model.predict(x_test_tfidf)

#step 8: evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
