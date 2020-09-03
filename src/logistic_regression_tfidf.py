import pandas as pd

from nltk.tokenize import word_tokenize

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import linear_model
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # rezd the training data
    df = pd.read_csv("../input/imdb.csv")

    # map positive to 1 and negative to 0
    df.sentiment =  df.sentiment.map(lambda x: 1 if x== "positive" else 0)

    # Creat stratified folds
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f


    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        valid_df = df[df.kfold == fold_].reset_index(drop=True)

        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
        tfidf_vec.fit(train_df.review)
        X_train, y_train = tfidf_vec.transform(train_df.review), train_df.sentiment
        X_valid, y_valid = tfidf_vec.transform(valid_df.review), valid_df.sentiment

        model = linear_model.LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        accuracy = metrics.accuracy_score(y_valid, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
