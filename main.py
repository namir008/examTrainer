import numpy as np
import pandas as pd
import os

from sklearn import model_selection as sk_model_selection
from sklearn.feature_extraction import text as sk_fe_text
from sklearn import svm as sk_svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as sk_metrics

#path for datasets
base_dir = 'venv/dataset/'
df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
df_submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

df_train.head()

X_train = df_train["question"]
y_train = df_train["topic"].values

#this blocks commonly used words
tfidf = sk_fe_text.TfidfVectorizer(stop_words = 'english')
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)

kn = KNeighborsClassifier()

parameters = {
    'n_neighbors' : [5, 25],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

model = sk_svm.SVC(
    kernel='rbf',
    class_weight='balanced',
    random_state=42,
)

model = sk_model_selection.GridSearchCV(
    model,
    estimator = kn,
    param_grid = parameters,
    cv=5,
    scoring='f1',
    n_jobs=-1,
)

model.fit(X_train, y_train)

print(f'Best parameters: {model.best_params_}')
print(f'Mean cross-validated F1 score of the best_estimator: {model.best_score_:.3f}')

X_test = df_test["question"]
X_test = tfidf.transform(X_test)
y_test_pred = model.predict(X_test)

df_submission["topic"] = y_test_pred
df_submission.to_csv("submission.csv",index=False)