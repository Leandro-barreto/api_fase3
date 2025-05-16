# 📥 API de Download dos Dados ANEEL

Este módulo contém uma API FastAPI que faz o download do conjunto de dados de projetos regulados pela ANEEL e os salva em um banco SQLite.
Também é possível rodar o arquivo local_download.py que baixa os arquivos sem o uso da API.

Documentação da API pelo swagger

## 🚀 Funcionalidades

- Faz web scraping dos dados abertos no portal da ANEEL
- Constrói e armazena a tabela `aneel_202505` no SQLite
- Salvas arquivos csv localmente

## 📁 Estrutura

```bash
download_api/
├── routers              # Rota para download dos dados
│   ├── __init__.py
│   └── donwloaddata.py
├── services             # Funções auxiliares
│   ├── __init__.py
│   └── databases.py
├── main.py              # API principal FastAPI
├── local_download.py    # Script para baixar os dados sem o uso da API.
└── aneel_ped.db         # Arquivo SQLite gerado (após execução)
```

## 🛠️ Como Executar

```bash
uvicorn main:app --reload
```

Acesse: [http://localhost:8000/docs](http://localhost:8000/docs)

## Requisitos

```bash
pip install fastapi uvicorn pandas sqlite3
```