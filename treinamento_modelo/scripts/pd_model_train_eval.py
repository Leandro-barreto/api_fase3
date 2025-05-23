import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(df):
    df = df.drop(columns=['meses_duracao', 'custo_total'])
    corr_matrix = df.corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f")
    plt.title('Correlation Matrix')

    plt.savefig('assets/correlation_matrix.png', dpi=300, bbox_inches='tight') 

    plt.close()


def calcula_mostra_matriz_confusao(y_true, y_pred, normalize=False, percentage=True, tipo='Treino'):
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

    matriz_confusao = {
        'True Positive': f"{cm[0,0]}",
        'False Positive': f"{cm[0,1]}",
        'False Negative': f"{cm[1,0]}",
        'True Negative': f"{cm[1,1]}",
    }
    with open(f"assets/matriz_confusao_{tipo.lower()}.json", 'w') as file:
        json.dump(matriz_confusao, file, indent=4)


def eval_model(model, X, y, tipo=""):
    y_pred = model.predict(X)

    print("Gradient Boosting Classifier")
    print("=" * 40)
    print(f"Dados de {tipo}")
    print("=" * 40)
    print("Matriz de Confusão")
    print("-" * 40)
    calcula_mostra_matriz_confusao(y, y_pred, normalize=False, tipo=tipo)
    print("-" * 40)
    print("Métricas")
    print("-" * 40)
    print("> Globais <")
    print(f"Acurácia: {accuracy_score(y, y_pred):.4f}")
    print(f"Precision (ponderado): {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall (ponderado): {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 (ponderado): {f1_score(y, y_pred, average='weighted'):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y, y_pred, average='macro'):.4f}")
    print()
    global_metrics = {
        "Acuracia": f"{accuracy_score(y, y_pred):.4f}",
        "Precision (ponderado)": f"{precision_score(y, y_pred, average='weighted'):.4f}",
        "Recall (ponderado)": f"{recall_score(y, y_pred, average='weighted'):.4f}",
        "F1 (ponderado)": f"{f1_score(y, y_pred, average='weighted'):.4f}",
        "AUC-ROC": f"{roc_auc_score(y, y_pred, average='macro'):.4f}",
    }
    print("> Por label <")
    print(classification_report(y, y_pred, labels=[1, 0], target_names=["Sucesso", "Falha"]))
    with open(f"assets/metricas_globais_{tipo.lower()}.json", 'w') as file:
        json.dump(global_metrics, file, indent=4)


def model_train(X_train, y_train):
    gbc = GradientBoostingClassifier(random_state=8)

    param_grid_gbc = {
        "n_estimators": [10, 20, 30],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2]
    }

    cv_gbc = GridSearchCV(
        gbc,
        param_grid_gbc,
        cv=5,
        scoring='f1_weighted'
    )
    cv_gbc.fit(X_train, y_train)
    return cv_gbc.best_estimator_


def model_save(df, label_col='label', model_path='model.pkl'):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    print('Divisão entre treino e teste:')
    print(f'Treino {X_train.shape[0]}')
    print(X_train)
    print(f'Teste {X_test.shape[0]}')
    correlation_matrix(X_train)

    modelo = model_train(X_train, y_train)
    eval_model(modelo, X_test, y_test, "Teste")

    joblib.dump(modelo, model_path)
    return f"Modelo salvo em: {model_path}"
