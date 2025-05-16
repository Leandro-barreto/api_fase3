from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
from services.functions import read_uploaded_file, apply_model, render_result_table

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Erro ao ler o arquivo: {str(e)}</h3>", status_code=400)

    try:
        df_with_preds = apply_model(df, model_path = 'modelo_classificacao.pkl')
    except Exception as e:
        return HTMLResponse(content=f"<h3>{str(e)}</h3>", status_code=500)

    html_content = render_result_table(df_with_preds, file.filename)
    return HTMLResponse(content=html_content)
