from fastapi import UploadFile
import pandas as pd
import io
from services import inference

def check_cols(df):
    cols_to_check = [
        'AnoCadastroPropostaProjeto',
        'DatConclusaoProjeto',
        'DatGeracaoConjuntoDados',
        'DscChamPEDEstrategico',
        'DscCodProjeto',
        'DscTituloProjeto',
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
    if 'IdcSituacaoProjeto' in cols_df:
        cols_df.remove('IdcSituacaoProjeto')
    list_val = list(set(cols_to_check) - set(cols_df))
    if len(list_val) > 0:
        print('Aqui')
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
    if 'IdcSituacaoProjeto' in df.columns:
        df = df.drop(columns='IdcSituacaoProjeto')
    check_cols(df)
    return df

def apply_model(df: pd.DataFrame, model_path) -> pd.DataFrame:
    try:
        df_results = inference.inference(df, model_path)
        return df_results
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
