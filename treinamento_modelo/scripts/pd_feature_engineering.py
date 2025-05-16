import pandas as pd
import re
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

# 1. Leitura do SQLite
def read_from_db(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM aneel_202505", conn)
    conn.close()
    return df

# 2. Remoção de colunas
def remove_cols(df):
    colunas_para_remover = [
        "AnoCadastroPropostaProjeto", "DatConclusaoProjeto", "DatGeracaoConjuntoDados",
        "DscChamPEDEstrategico", "DscCodProjeto", "NomAgente", "NumCPFCNPJ", "SigAgente",
        "VlrCustoTotalAuditado", "DscTituloProjeto"
    ]
    return df.drop(columns=colunas_para_remover, errors="ignore")

# 3. Renomear colunas
def renaming_cols(df):
    renomear = {
        "IdcSituacaoProjeto": "status",
        "SigFasInovacaoProjeto": "cadeia_inovacao",
        "SigSegmentoSetorEletrico": "segmento_setor",
        "SigTemaProjeto": "tema",
        "SigTipoProdutoProjeto": "produto",
        "QtdMesesDuracaoPrevista": "meses_duracao",
        "VlrCustoTotalPrevisto": "custo_total",
        "_id": "id"
    }
    return df.rename(columns=renomear)

# 4. Conversão de tipos
def cast_types(df):
    df["meses_duracao"] = pd.to_numeric(df["meses_duracao"], errors="coerce").astype("Int64")
    df["custo_total"] = pd.to_numeric(df["custo_total"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df

# 5. Normalizar caracteres
def normalize_text(text):
    if pd.isnull(text):
        return None
    text = str(text).lower()
    text = re.sub(r'[àáâãäå]', 'a', text)
    text = re.sub(r'[èéêë]', 'e', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[òóôõö]', 'o', text)
    text = re.sub(r'[ùúûü]', 'u', text)
    text = re.sub(r'[ç]', 'c', text)
    text = re.sub(r'[ýÿ]', 'y', text)
    text = re.sub(r'[ñ]', 'n', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.upper()

def remove_special_chars(df):
    df['status'] = df['status'].apply(normalize_text)
    return df

# 6. Remover linhas nulas ou com status inválido
def remove_lines(df):
    df = df.dropna()
    df = df[df["status"].isin(["CONCLUIDO", "CANCELADO"])]
    return df

# 7. One-hot encoding
def one_hot_encoding(df, cols_to_encode):
    return pd.get_dummies(df, columns=cols_to_encode, prefix_sep='_', dummy_na=False)

# 8. Escalonamento Min-Max
def scaling_numbers(df, cols_to_scale):
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

# 9. Balanceamento de classes (undersampling)
def balancing_classes(df, target_col='status'):
    count_min = df[target_col].value_counts().min()
    df_0 = df[df[target_col] == 0]
    df_1 = df[df[target_col] == 1]
    df_0_under = resample(df_0, replace=False, n_samples=count_min, random_state=8)
    df_1_under = resample(df_1, replace=False, n_samples=count_min, random_state=8)
    return pd.concat([df_0_under, df_1_under]).sample(frac=1, random_state=8).reset_index(drop=True)

# 10. Pipeline completo
def dataset_prep(df):
    df = remove_cols(df)
    df = renaming_cols(df)
    df = cast_types(df)
    df = remove_special_chars(df)
    df = remove_lines(df)

    df['status'] = df['status'].map({"CONCLUIDO": 1, "CANCELADO": 0})
    df = balancing_classes(df, target_col='status')

    df = one_hot_encoding(df, cols_to_encode=['cadeia_inovacao', 'segmento_setor', 'tema', 'produto'])
    df = scaling_numbers(df, cols_to_scale=["meses_duracao", "custo_total"])

    df = df.drop(columns=["id"], errors="ignore")  # opcional

    X = df.drop(columns=["status"])
    y = df["status"]
    return X, y
