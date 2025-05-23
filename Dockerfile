# Base com Python
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copia pasta aplicativos
COPY app_infer/. .
COPY fly.toml .
COPY Dockerfile .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta da FastAPI (usando Uvicorn) 
EXPOSE 8000

# Comando de execução
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
