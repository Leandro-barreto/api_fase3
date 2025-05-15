from fastapi import UploadFile
import pandas as pd
import io
from pyspark.sql import SparkSession
from services import inference

spark = SparkSession.builder.appName("AneelData").config("spark.ui.showConsoleProgress", "true").getOrCreate()

def check_cols(df):
    cols_to_check = [
        'AnoCadastroPropostaProjeto',
        'DatConclusaoProjeto',
        'DatGeracaoConjuntoDados',
        'DscChamPEDEstrategico',
        'DscCodProjeto',
        'DscTituloProjeto',
        'IdcSituacaoProjeto',
        'NomAgente',
        'NumCPFCNPJ',
        'QtdMesesDuracaoPrevista',
        'SigAgente',
        'SigFasInovacaoProjeto',
        'SigSegmentoSetorEletrico',
        'SigTemaProjeto',
        'SigTipoProdutoProjeto',
        'VlrCustoTotalAuditado',
        'VlrCustoTotalPrevisto',
        '_id'
        ]
    cols_df = df.columns.tolist()
    list_val = list(set(cols_df) - set(cols_to_check))
    if len(list_val) > 0:
        raise ValueError(f"Arquivo escolhido não contém as colunas corretas: {', '.join(list_val)}")
    return 0

def formatar_cnpj_serie(coluna):
    return coluna.astype(str).str.zfill(14).str.replace(
        r"(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})",
        r"\1.\2.\3/\4-\5",
        regex=True
    )

def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    elif filename.endswith(".json"):
        df =  pd.read_json(io.StringIO(contents.decode("utf-8")))
    else:
        raise ValueError("Formato de arquivo não suportado. Use .csv ou .json.")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    check_cols(df)
    df = df[~df['NumCPFCNPJ'].isna()]
    df['NumCPFCNPJ'] = df['NumCPFCNPJ'].astype(int)
    df["NumCPFCNPJ"] = formatar_cnpj_serie(df["NumCPFCNPJ"])
    return df

def apply_model(df: pd.DataFrame, model_path) -> pd.DataFrame:
    try:
        df_spark = spark.createDataFrame(df)
        df_results = inference.inference(df_spark, model_path)
        return df_results[['NumCPFCNPJ', 'probability', 'prediction']]
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
