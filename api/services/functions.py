from fastapi import UploadFile
import pandas as pd
import io

def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(io.StringIO(contents.decode("utf-8")))
    elif filename.endswith(".json"):
        return pd.read_json(io.StringIO(contents.decode("utf-8")))
    else:
        raise ValueError("Formato de arquivo não suportado. Use .csv ou .json.")

def apply_model(df: pd.DataFrame, model) -> pd.DataFrame:
    try:
        predictions = model.predict(df)
        df["predicao"] = predictions
        return df
    except Exception as e:
        raise RuntimeError(f"Erro ao aplicar o modelo: {e}")

def render_result_table(df: pd.DataFrame, filename: str) -> str:
    table_html = df.to_html(index=False)
    return f"""
    <h3>Resultados para o arquivo: {filename}</h3>
    <div>{table_html}</div>
    <br>
    <a href="/">Voltar</a>
    """
