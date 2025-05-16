import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql.functions import col, udf, lit
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession.builder.appName("AneelData").getOrCreate()

def remove_cols(df):
    colunas_para_remover = ["AnoCadastroPropostaProjeto",
                        "DatConclusaoProjeto",
                        "DatGeracaoConjuntoDados",
                        "DscChamPEDEstrategico",
                        "DscCodProjeto",
                        "NomAgente",
                        "NumCPFCNPJ",
                        "SigAgente",
                        "VlrCustoTotalAuditado",
                        "DscTituloProjeto"]

    df = df.drop(*colunas_para_remover)
    return df

def renaming_cols(df):
    df = df.withColumnRenamed("IdcSituacaoProjeto", "status")
    df = df.withColumnRenamed("SigFasInovacaoProjeto", "cadeia_inovacao")
    df = df.withColumnRenamed("SigSegmentoSetorEletrico", "segmento_setor")
    df = df.withColumnRenamed("SigTemaProjeto", "tema")
    df = df.withColumnRenamed("SigTipoProdutoProjeto", "produto")
    df = df.withColumnRenamed("QtdMesesDuracaoPrevista", "meses_duracao")
    df = df.withColumnRenamed("VlrCustoTotalPrevisto", "custo_total")
    df = df.withColumnRenamed("_id", "id")
    return df

def cast_types(df):
    df = df.withColumn("VlrCustoTotalPrevisto", col("VlrCustoTotalPrevisto").cast(IntegerType()))
    df = df.withColumn("QtdMesesDuracaoPrevista", col("QtdMesesDuracaoPrevista").cast(IntegerType()))
    df = df.withColumn("_id", col("_id").cast(IntegerType()))
    return df

def remove_special_chars(df):
    def normalize_text(text):
        if text is None:
            return None
        text = str(text)
        text = re.sub(r'[àáâãäåÀÁÂÃÄÅ]', 'a', text)
        text = re.sub(r'[èéêëÈÉÊË]', 'e', text)
        text = re.sub(r'[ìíîïÌÍÎÏ]', 'i', text)
        text = re.sub(r'[òóôõöòóôõöÒÓÔÕÖ]', 'o', text)
        text = re.sub(r'[ùúûüÙÚÛÜ]', 'u', text)
        text = re.sub(r'[çÇ]', 'c', text)
        text = re.sub(r'[ýÿÝŸ]', 'y', text)
        text = re.sub(r'[ñÑ]', 'n', text)
        text = re.sub(r'[^\w\s]', '', text) # Remove apóstrofos e outros caracteres especiais
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-+', '', text)
        text = text.upper()
        return text

    normalize_udf = udf(normalize_text, StringType())
    df = df.withColumn("status", normalize_udf(col("status")))
    return df


def one_hot_encoding_spark(df: DataFrame, cols_to_apply: list) -> DataFrame:

    for col in cols_to_apply:
        pivot_df = df.groupBy('id').pivot(col).agg(lit(1)).na.fill(0)
        distinct_values = pivot_df.columns
        distinct_values.remove('id')
        if 'null' in distinct_values:
            distinct_values.remove('null')
            pivot_df = pivot_df.drop('null')

        df = df.join(pivot_df, 'id', how='inner')

        # for val in distinct_values:
        #     df = df.withColumnRenamed(val, f"{col}_{val}")
        df = df.drop(col)

    
    return df

def scaling_numbers(df, cols_to_scale = ["meses_duracao", "custo_total"]):
    for col in cols_to_scale:
        # Vetoriza as colunas numéricas
        assembler_meses = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        df = assembler_meses.transform(df)

        # Cria o scaler
        scaler_meses = MinMaxScaler(inputCol=f"{col}_vec", outputCol=f"{col}_scaled")

        # Aplica o scaler
        scaler_model_meses = scaler_meses.fit(df)
        df = scaler_model_meses.transform(df)

        # Drop nas colunas criadas para fazer a transformação
        df = df.drop(col, f"{col}_vec")

    return  df

def dataset_prep(df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    dataset_prep = assembler.transform(df).select('id', 'features') 
    return dataset_prep

def fill_missing_cols(df, feature_cols):
    missing_cols = list(set(feature_cols) - set(df.columns))
    for col in missing_cols:
        df = df.withColumn(col, lit(0))
    return df

def create_feat_df(df):
    feature_cols = ['CS', 'DE', 'IM', 'LP', 'PA', 'PB', 'C', 'D', 'G', 'T', 'EE', 'FA', 'GB', 'GT', 'MA',
                    'MF', 'OP', 'OU', 'PL', 'QC', 'SC', 'SE', 'CD', 'CM', 'ME', 'MS', 'SM', 'SW',
                    'meses_duracao_scaled', 'custo_total_scaled']
    
    df = remove_cols(df)
    df = cast_types(df)
    df = renaming_cols(df)
    df = remove_special_chars(df)
    df = one_hot_encoding_spark(df, cols_to_apply = ['cadeia_inovacao', 'segmento_setor', 'tema', 'produto'])
    df = df.fillna(0, subset=["meses_duracao", "custo_total"])
    df = scaling_numbers(df, cols_to_scale = ["meses_duracao", "custo_total"])
    df = fill_missing_cols(df, feature_cols)
    df = dataset_prep(df, feature_cols=feature_cols)

    return df

def inference(df, model_path):
    # df_id_cnpj = df.select(["_id", "NumCPFCNPJ"]) 
    # df_id_cnpj = df_id_cnpj.withColumnRenamed('_id', 'id')
    df_features = create_feat_df(df)
    loaded_model = RandomForestClassificationModel.load(model_path)
    df_results = loaded_model.transform(df_features)
    # df_results = df_id_cnpj.join(df_results, on='id', how='inner').toPandas()
    return df_results.toPandas()


