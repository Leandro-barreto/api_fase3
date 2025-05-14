from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import joblib  # ou pickle, se estiver usando scikit-learn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Carrega modelo previamente treinado
model = joblib.load("modelo_classificacao.pkl")  # substitua pelo seu caminho

@app.get("/", response_class=HTMLResponse)
async def main_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif filename.endswith(".json"):
            df = pd.read_json(io.StringIO(contents.decode("utf-8")))
        else:
            return HTMLResponse(content=f"<h3>Formato de arquivo não suportado: {filename}</h3>", status_code=400)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Erro ao ler o arquivo: {str(e)}</h3>", status_code=400)

    try:
        # 🔍 Aplica o modelo (certifique-se de que o modelo aceita essas colunas)
        preds = model.predict(df.drop(columns='target'))
        df["predicao"] = preds
    except Exception as e:
        return HTMLResponse(content=f"<h3>Erro ao aplicar o modelo: {str(e)}</h3>", status_code=500)

    # Converte DataFrame para HTML
    result_table = df.to_html(index=False)

    return HTMLResponse(content=f"""
    <h3>Resultados para o arquivo: {filename}</h3>
    <div>{result_table}</div>
    <br>
    <a href="/">Voltar</a>
    """)
