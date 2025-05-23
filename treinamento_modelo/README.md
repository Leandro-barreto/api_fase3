# 🧠 Treinamento do Modelo de Classificação

Este módulo realiza todo o pipeline de preparação, balanceamento, treinamento e avaliação de um modelo de classificação de projetos da ANEEL com scikit-learn.

## ⚙️ Funcionalidades

### Análises exploratórias em:

[Google Colab - Versão Pandas](https://colab.research.google.com/drive/18X2YVaqOE6n7Cl4u_2zDTZkC9JuG074X?authuser=1#scrollTo=0FcwcWdBZEjW)


[Google Colab - 1a versão em spark](https://colab.research.google.com/drive/12d79b2pbGaN5qZdBNgDkU23CdyqoX6E_?usp=sharing)


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

## 📊 Desempenho do Modelo (Dados de Teste)

### Métricas Globais

| Métrica                | Valor   |
|------------------------|---------|
| Acurácia               | 0.8071  |
| Precisão (ponderado)   | 0.8131  |
| Recall (ponderado)     | 0.8071  |
| F1 Score (ponderado)   | 0.8056  |
| AUC-ROC                | 0.8045  |

### Matriz de Confusão

| Real \ Previsto | Sucesso | Falha |
|-----------------|---------|-------|
| **Sucesso**     |    118  |   45  |
| **Falha**       |    20   |   154 |


#### As métricas estão salvas na pasta assets.

## 🛠️ Como Executar

```python
python pd_main.py
```

## Requisitos

```bash
pip install pandas scikit-learn joblib seaborn matplotlib
```
