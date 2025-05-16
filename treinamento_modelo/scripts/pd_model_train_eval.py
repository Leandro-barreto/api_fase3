import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import joblib


def calcula_mostra_matriz_confusao(y_true, y_pred, normalize=False, percentage=True):
    labels = [1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / row_sums
        if percentage:
            cm *= 100
        cm = cm.astype(int)

    print(" " * 20 + "Previsto")
    print(" " * 15 + "Sucesso" + " " * 5 + "Falha")
    print(" " * 4 + "Sucesso" + " " * 6 + f"{cm[0,0]}" + " " * 7 + f"{cm[0,1]}")
    print("Real")
    print(" " * 4 + "Falha" + " " * 9 + f"{cm[1,0]}" + " " * 7 + f"{cm[1,1]}")


def eval_model(model, X, y, tipo=""):
    y_pred = model.predict(X)

    print("Random Forest Classifier")
    print("=" * 40)
    print(f"Dados de {tipo}")
    print("=" * 40)
    print("Matriz de Confusão")
    print("-" * 40)
    calcula_mostra_matriz_confusao(y, y_pred, normalize=False)
    print("-" * 40)
    print("Métricas")
    print("-" * 40)
    print("> Globais <")
    print(f"Acurácia: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision (ponderado): {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall (ponderado): {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 (ponderado): {f1_score(y, y_pred, average='weighted'):.4f}")
    print()
    print("> Por label <")
    print(classification_report(y, y_pred, labels=[1, 0], target_names=["Sucesso", "Falha"]))


def model_train(X_train, y_train):
    rfc = RandomForestClassifier(random_state=8)

    param_grid = {
        "n_estimators": [10, 20, 30],
        "max_depth": [5, 10, 15]
    }

    cv = GridSearchCV(rfc, param_grid, cv=5, scoring='f1_weighted')
    cv.fit(X_train, y_train)

    return cv.best_estimator_


def model_save(df, label_col='label', model_path='model.pkl'):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

    modelo = model_train(X_train, y_train)

    eval_model(modelo, X_train, y_train, "Treino")
    eval_model(modelo, X_test, y_test, "Teste")

    joblib.dump(modelo, model_path)
    return f"Modelo salvo em: {model_path}"
