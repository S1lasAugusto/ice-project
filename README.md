# Projeto de Análise de Texto

Este é um projeto em Python para análise de texto que utiliza embeddings pré-treinados do FastText e do GloVe. 

## Requisitos

Antes de executar este projeto, é necessário baixar os embeddings pré-treinados e colocá-los na pasta do projeto. Você pode obter os embeddings nos seguintes links:

- [FastText English Vectors](https://fasttext.cc/docs/en/english-vectors.html): Faça o download do arquivo `wiki-news-300d-1M.vec.zip` de embedding.
- [GloVe](https://nlp.stanford.edu/projects/glove/): Faça o download do arquivo `glove.6B.zip` e extraia o arquivo de embedding de 300 dimensões.

Além disso, você também precisará do dataset para análise de texto. Você pode encontrá-lo [aqui](https://www.kaggle.com/datasets/thevirusx3/automated-essay-scoring-dataset/code).

Também é necessário baixar o submódulo PonyGE2, que será utilizado para a evolução gramatical. Para baixa-lo, basta executar o seguinte comando:
```bash
git submodule update --recursive
```

## Configuração do Projeto

Após baixar os embeddings e o dataset, coloque-os na pasta **./embeddings**.

## Executando o Projeto

Certifique-se de que os embeddings estão na pasta **./embeddings** e execute os scripts Python fornecidos.

```bash
CE2VA_model.ipynb
```
