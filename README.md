# Classificador de Projetos ANEEL

Este é um projeto completo para classificação de projetos da ANEEL com FastAPI, scikit-learn e SQLite, dividido em três módulos principais:

- API para download de dados públicos da ANEEL
- Pipeline de preparação e treino de modelo de classificação
- Aplicativo de inferência implantável no Fly.io com interface web

## 🚀 Funcionalidades

- 🔽 Download automatizado de dados ANEEL via API
- ⚙️ Processamento e engenharia de features com Pandas
- 🧠 Treinamento de modelo Random Forest com GridSearchCV
- 📈 Avaliação com métricas e matriz de confusão
- 📤 Interface de upload de arquivos e resposta com predições

## 📁 Estrutura do Projeto

```bash
projeto/
├── app_infer/              # Aplicação FastAPI para predição (Fly.io)
├── download_api/           # API de download de dados da ANEEL
├── treinamento_modelo/     # Treinamento e avaliação do modelo
└── README.md               # Este arquivo
```

## 🛠️ Como Executar

1. Execute a API de download: `download_api/`
2. Treine o modelo com os dados: `treinamento_modelo/`
3. Suba a aplicação de inferência: `app_infer/`

## Arquivos auxiliares

1. Os dicionário de dados está em dd-projetos-de-pd-em-energia-eletrica.pdf
2. Os arquivos aneel.csv, aneel.json e aneel_min.json, são arquivos para subir na aplicação de inferência