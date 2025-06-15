import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nombre_job", type=str, default="ClasificacionTextoNB")
    parser.add_argument("--alpha_list", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    return parser.parse_args()

def load_and_prepare_data():
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train = vectorizer.fit_transform(train.data)
    y_train = train.target
    X_test = vectorizer.transform(test.data)
    y_test = test.target

    return X_train, X_test, y_train, y_test, train.target_names

def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, target_names, alpha_list):
    mlflow.set_experiment(nombre_job)
    for alpha in alpha_list:
        with mlflow.start_run():
            model = MultinomialNB(alpha=alpha)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            mlflow.log_param("alpha", alpha)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "modelo_nb")

            print(f"Alpha: {alpha}")
            print(classification_report(y_test, y_pred, target_names=target_names))
