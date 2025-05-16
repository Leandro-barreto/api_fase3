from scripts import feature_engineering, model_train_eval

def create_model():
    df = feature_engineering.create_feat_df()
    print('Inicio do Treinamento')
    print(model_train_eval.model_save(df))
    return 0

if __name__=='__main__':
    create_model()
