# desafio-estag-2022
## Instruções de instalação - Windows

Para a criação do ambiente virtual utilizaremos Anaconda. A implementação só utiliza o Tensorflow como pacote portanto, ao abrir o Anaconda Prompt e criar o ambiente virtual é necessário adicionar somente o pacote do Tensorflow:

``
conda create -n estagiopetrecml python tensorflow
``

## Instruções de uso

Com o ambiente virtual criado e o repositório com a solução clonado basta ativar o ambiente no Anaconda Prompt para replicar a solução:

``
conda activate estagiopetrecml
``

A solução foi separada em treinamento e avaliação. Portanto, inicialmente é necessário treinar o modelo com as imagens MNIST e salvar o modelo treinado. Ainda no Anaconda Prompt:

``
python treinamento.py
``

Por fim, podemos avaliar o modelo com o conjunto de teste:

``
python avaliacao.py
``
