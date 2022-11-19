######### Bibliotecas #########
import tensorflow as tf


######### Definição da Rede Neural #########
def createMNISTModel(mnistShape, numberOfClasses):

    # Rede Neural para a classificação dos números do conjunto MNIST #
    # 1 camada de normalização: [0, 255] -> [0, 1] #
    # 2 camadas de 96 neurônios #
    # 1 camada para a classificação final: 10 neurônios p/ os 10 números e,
    # ativação 'softmax' para distribuição de probabilidade entre as categorias (números) #
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=mnistShape),
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dense(numberOfClasses, activation='softmax')
    ])

    # Definição de métricas da rede neural #
    # Métricas: Acurácia, Recall e Precisão #
    # A métrica F1 Score pode ser calculada a partir de Recall e Precisão #
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
    )

    # Qntd de parâmetros treináveis #
    trainableParams = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print("Numero de parametros treinaveis:",trainableParams)

    return model



def main():
    ######### Dataset p/ Treino ##########
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # One-hot encoding para possibilitar o uso das métricas Recall e Precision ao compilar o modelo #
    y_train = tf.one_hot(y_train,10)

    # Infos do dataset MNIST #
    mnistLabels = [0,1,2,3,4,5,6,7,8,9]
    mnistShape = x_train[0].shape
    numberOfClasses = len(mnistLabels)

    print("Formato das imagens MNIST:", mnistShape)
    print("Qntd de imagens p/ treino: ", len(x_train))
    # print("Qntd de imagens p/ teste:", len(x_test))

    ######### Rede Neural #########
    # Criação e treinamento da Rede Neural #
    mnistModel = createMNISTModel(mnistShape, numberOfClasses)

    mnistModel.fit(
        x = x_train,
        y = y_train,
        epochs = 10
    )

    # Salvando o modelo treinado #
    mnistModel.save_weights('MNISTmodel.h5')
    print('Modelo Salvo!')

    return 0

if __name__ == "__main__":
    main()