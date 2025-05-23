import pandas as pd
import re
import joblib
from sklearn.preprocessing import MinMaxScaler

# 1. Remoção de colunas irrelevantes
def remove_cols(df):
    colunas_para_remover = [
        "AnoCadastroPropostaProjeto", "DatConclusaoProjeto", "DatGeracaoConjuntoDados",
        "DscChamPEDEstrategico", "DscCodProjeto", "NomAgente", "NumCPFCNPJ", "SigAgente",
        "VlrCustoTotalAuditado", "DscTituloProjeto"
    ]
    return df.drop(columns=colunas_para_remover, errors="ignore")

# 2. Renomear colunas
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

# 3. Conversão de tipos
def cast_types(df):
    df["meses_duracao"] = pd.to_numeric(df["meses_duracao"], errors="coerce").fillna(0).astype("float")
    df["custo_total"] = pd.to_numeric(df["custo_total"], errors="coerce").fillna(0).astype("float")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df

# 4. Limpeza de texto
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
    if "status" in df.columns:
        df["status"] = df["status"].apply(normalize_text)
    return df

# 5. One-hot encoding (igual ao treino)
def one_hot_encoding(df, cols_to_encode):
    return pd.get_dummies(df, columns=cols_to_encode, prefix_sep='_', dummy_na=False)

# 6. Normalização Min-Max
def scaling_numbers(df, cols_to_scale):
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

# 7. Garantir que todas as colunas esperadas existam
def fill_missing_cols(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

# 8. Pipeline de preparação de features para inferência
def create_feat_df(df, feature_cols):
    df = remove_cols(df)
    df = renaming_cols(df)
    df = cast_types(df)
    df = remove_special_chars(df)
    df = one_hot_encoding(df, ['cadeia_inovacao', 'segmento_setor', 'tema', 'produto'])
    df = scaling_numbers(df, ["meses_duracao", "custo_total"])
    df_feat = fill_missing_cols(df, feature_cols)
    return df_feat, df.get("id", pd.Series(range(len(df))))

# 9. Função principal de inferência
def inference(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    df_features, ids = create_feat_df(df, feature_cols=model.feature_names_in_)
    # Previsão
    preds = model.predict(df_features)

    # Probabilidade da classe 1 (sucesso)
    probs = model.predict_proba(df_features)[:, 1]

    # Retorno com colunas adicionais
    df_result = df.copy()
    df_result["predicao"] = preds
    df_result["prob_sucesso"] = probs
    df_result['desc_new'] = df_result['DscTituloProjeto'].str[:500]
    return df_result[["_id", "desc_new", "predicao"]]
