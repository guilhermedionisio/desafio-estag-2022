import tensorflow as tf
from treinamento import createMNISTModel

def main():
    ######### Dataset p/ teste ##########
    _ , (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    mnistShape = x_test[0].shape
    numberOfClasses = 10
    
    # One-hot encoding para possibilitar o uso das métricas Recall e Precision ao compilar o modelo #
    y_test = tf.one_hot(y_test,10)

    ######### Rede Neural #########
    # Criando um novo modelo e carregando os parâmetros já treinados #
    savedMnistModel = createMNISTModel(mnistShape, numberOfClasses)
    savedMnistModel.load_weights('MNISTmodel.h5')
    print('Modelo carregado!')

    # Avaliar as métricas definidas no modelo #
    _, testAccuracy, testRecall, testPrecision = savedMnistModel.evaluate(x_test, y_test)
    testF1Score = 2*(testRecall * testPrecision)/ (testRecall + testPrecision)

    # Armazenas as previsões do modelo no conjunto de teste para a criação da matriz de confusão #
    y_pred = savedMnistModel.predict(x_test)
    y_pred = [y_pred[idx].argmax() for idx in range(0, 10000)]
    y_test = [y_test[idx].numpy().argmax() for idx in range(0, 10000)]

    # Criação da matriz de confusão #
    # Normalização das linhas para obter os valores percentuais: ( a1n / sum(a11...a1n) ) #
    confusionMatrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred, num_classes=numberOfClasses)
    confusionMatrixPercentage = confusionMatrix/tf.reduce_sum(confusionMatrix, axis=1)
    confusionMatrixPercentage = confusionMatrixPercentage.numpy().round(decimals = 2)

    # Apresentação das métricas #
    print("Precisão no conjunto de teste:", round(testPrecision, 3)*100, "%")
    print("Recall no conjunto de teste:", round(testRecall, 3)*100, "%")
    print("F1 Score no conjunto de teste:", round(testF1Score, 3)*100, "%")
    print("Acurácia no conjunto de teste:", round(testAccuracy, 3)*100, "%")

if __name__ == "__main__":
    main()