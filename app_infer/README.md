# 🌐 Aplicativo de Inferência (FastAPI + Fly.io)

Este módulo contém uma aplicação FastAPI pronta para deploy no Fly.io. Ele permite enviar arquivos `.csv` ou `.json` com novos projetos da ANEEL e retorna as predições de sucesso e a probabilidade associada.

## 🚀 Funcionalidades

- Upload de arquivos pela interface web
- Processamento e validação dos dados
- Predição com modelo `.pkl` treinado
- Cálculo da probabilidade de sucesso
- Interface HTML simples e funcional

## 📁 Estrutura

```bash
app_infer/
├── services                    # Funções auxiliares
│   ├── __init__.py
│   ├── inference.py            # Pipeline de preparação e inferência
│   └── functions.py
├── main.py                     # API FastAPI com interface HTML
├── modelo_classificacao.pkl    # Modelo treinado (importado)
└── templates/index.html        # Interface web de upload
```

## 🛠️ Como Executar Localmente

```bash
uvicorn main:app --reload
```

Acesse em: [http://localhost:8000](http://localhost:8000)

## 🚀 Deploy no Fly.io

Foi realizado deploy em uma aplicação no Fly.io para acessar entrar em:
[https://api-fase3.fly.dev/](hhttps://api-fase3.fly.dev/)

## Requisitos

```bash
pip install fastapi uvicorn pandas scikit-learn joblib jinja2 python-multipart
```