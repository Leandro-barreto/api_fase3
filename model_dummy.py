# Treinar um modelo dummy para teste
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

iris = load_iris()

# Converte para DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Salva como CSV
df.to_csv("iris.csv", index=False)


X, y = load_iris(return_X_y=True)
# model = RandomForestClassifier().fit(X, y)
# joblib.dump(model, "modelo_classificacao.pkl")