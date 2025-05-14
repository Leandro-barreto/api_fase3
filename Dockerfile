# Base com Python
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copia tudo
COPY . .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta da FastAPI (usando Uvicorn) 
EXPOSE 8080

# Comando de execução
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
