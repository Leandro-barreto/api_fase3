from scripts import pd_feature_engineering, pd_model_train_eval

def create_model():
    df_raw = pd_feature_engineering.read_from_db("../download_api/aneel_ped.db")
    X, y = pd_feature_engineering.dataset_prep(df_raw)
    print(X.shape)
    print(X.head(10))
    print(y.head(10))

    # Treinar e salvar modelo
    df = X.copy()
    df["label"] = y
    pd_model_train_eval.model_save(df, label_col="label", model_path="../app_infer/modelo_classificacao.pkl")

if __name__=='__main__':
    create_model()