# 🧠 Treinamento do Modelo de Classificação

Este módulo realiza todo o pipeline de preparação, balanceamento, treinamento e avaliação de um modelo de classificação de projetos da ANEEL com scikit-learn.

## ⚙️ Funcionalidades

- Leitura dos dados da base SQLite
- Limpeza, normalização e one-hot encoding
- Balanceamento com undersampling
- GridSearchCV com RandomForestClassifier
- Avaliação com matriz de confusão e métricas
- Salvamento do modelo em `.pkl`

## 📁 Estrutura

```bash
treinamento_modelo/
├── scripts                     # Funções auxiliares
│   ├── feature_engineering.py  # Pipeline de processamento de dados
│   └── model_train_eval.py     # Treinamento e avaliação
└── main.py                     # Script principal
```

## 🛠️ Como Executar

```python
python main.py
```

## Requisitos

```bash
pip install pandas scikit-learn joblib
```