from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def calcula_mostra_matriz_confusao(df_transform_modelo, normalize=False, percentage=True):
    tp = df_transform_modelo.select('label', 'prediction').where((col('label') == 1) & (col('prediction') == 1)).count()
    tn = df_transform_modelo.select('label', 'prediction').where((col('label') == 0) & (col('prediction') == 0)).count()
    fp = df_transform_modelo.select('label', 'prediction').where((col('label') == 0) & (col('prediction') == 1)).count()
    fn = df_transform_modelo.select('label', 'prediction').where((col('label') == 1) & (col('prediction') == 0)).count()

    valorP = 1
    valorN = 1

    if normalize:
        valorP = tp + fn
        valorN = fp + tn

    if percentage and normalize:
        valorP = valorP / 100
        valorN = valorN / 100

    print(' '*20, 'Previsto')
    print(' '*15, 'Sucesso', ' '*5, 'Falha')
    print(' '*4, 'Sucesso', ' '*6, int(tp/valorP), ' '*7, int(fn/valorP))
    print('Real')
    print(' '*4, 'Falha', ' '*9, int(fp/valorN), ' '*7, int(tn/valorN))

def eval_model(modelo_rfc, eval_data, tipo):
    previsoes_rfc_treino = modelo_rfc.transform(eval_data)

    evaluator = MulticlassClassificationEvaluator()

    print('Decision Tree Classifier')
    print("="*40)
    print(f"Dados de {tipo}")
    print("="*40)
    print("Matriz de Confusão")
    print("-"*40)
    calcula_mostra_matriz_confusao(previsoes_rfc_treino, normalize=False)
    print("-"*40)
    print("Métricas")
    print("-"*40)
    print("> Globais <")
    print("Acurácia: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "accuracy"}))
    print("Weighted Precision: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "weightedPrecision"}))
    print("Weighted Recall: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "weightedRecall"}))
    print("Weighted F1: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "weightedFMeasure"}))
    print("Weighted True Positive Rate: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "weightedTruePositiveRate"}))
    print("Weighted False Positive Rate: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "weightedFalsePositiveRate"}))
    print("Hamming Loss: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "hammingLoss"}))
    print("Log Loss: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "logLoss"}))
    print()
    print("> Por label <")
    print("Precisão Sucesso: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: 1}))
    print("Precisão Falha: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "precisionByLabel", evaluator.metricLabel: 0}))
    print("Recall Sucesso: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: 1}))
    print("Recall Falha: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "recallByLabel", evaluator.metricLabel: 0}))
    print("F1 Sucesso: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: 1}))
    print("F1 Falha: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "fMeasureByLabel", evaluator.metricLabel: 0}))
    print("True Positive Rate Sucesso: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "truePositiveRateByLabel", evaluator.metricLabel: 1}))
    print("True Positive Rate Falha: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "truePositiveRateByLabel", evaluator.metricLabel: 0}))
    print("False Positive Rate Sucesso: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "falsePositiveRateByLabel", evaluator.metricLabel: 1}))
    print("False Positive Rate Falha: %f" % evaluator.evaluate(previsoes_rfc_treino, {evaluator.metricName: "falsePositiveRateByLabel", evaluator.metricLabel: 0}))
    print("="*40)

def model_train(train_data):
    rfc = RandomForestClassifier(seed=8)

    # Tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rfc.numTrees, [10, 20, 30]) \
        .addGrid(rfc.maxDepth, [5, 10, 15]) \
        .build()

    # Validação cruzada
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedFMeasure")
    print('CV')
    cv = CrossValidator(estimator=rfc, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=2)

    modelo_rfc = cv.fit(train_data)
    return modelo_rfc

def model_save(dataset_prep, model_path = '../app_infer/model'):
    train_data, test_data = dataset_prep.randomSplit([0.75, 0.25], seed=8)
    modelo_rfc = model_train(train_data)
    eval_model(modelo_rfc, train_data, "Treino")
    eval_model(modelo_rfc, test_data, "Teste")

    # Salvando o melhor modelo
    modelo_rfc.bestModel.write().overwrite().save(model_path)
    return 'Modelo salvo'
