# Base com Python
FROM python:3.10-slim-bullseye

# Instala o Java (necessário para PySpark)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define o JAVA_HOME para o PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Diretório de trabalho
WORKDIR /app

# Copia pasta aplicativos
COPY app_infer/* .
COPY fly.toml .
COPY Dockerfile .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta da FastAPI (usando Uvicorn) 
EXPOSE 8000

# Comando de execução
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
