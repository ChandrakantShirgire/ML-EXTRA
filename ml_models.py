import re, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t).lower())
    t = re.sub(r"http\S+|#url_\w+#|#email_\w+#|#phone_\w+#", " ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", t)).strip()

df = pd.read_csv("fake_job_postings.csv")
df["text"] = (df.title.fillna("") + " " + df.description.fillna("") + " " + df.requirements.fillna("")).map(clean)
df = df[df.text.str.len() > 0]
df = df.groupby("fraudulent", group_keys=False).sample(frac=2000/len(df), random_state=42)
Xtr, Xte, ytr, yte = train_test_split(df.text, df.fraudulent, test_size=0.2, stratify=df.fraudulent, random_state=42)
tfidf = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3)
Xtr_v = tfidf.fit_transform(Xtr); Xte_v = tfidf.transform(Xte)

models = {"Naive Bayes": MultinomialNB(alpha=0.1),
          "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=10)}
rows, cms = {}, {}
for n, m in models.items():
    m.fit(Xtr_v, ytr); p = m.predict(Xte_v)
    rows[n] = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p), recall=recall_score(yte, p), f1=f1_score(yte, p))
    cms[n] = confusion_matrix(yte, p)
table = pd.DataFrame(rows).T.round(3)
table.to_csv("comparison_table.csv")
for n, c in cms.items(): print(n, "[[TN FP][FN TP]] =", c.tolist())
print(table)
print('Always-real accuracy:', round(1 - yte.mean(), 3))
