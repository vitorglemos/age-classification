# Age-Classification

# Introdução

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiJT2ISxJXw8c2XpHdw3Egx7QE72xuhZV2nB_PB306uefd98cnaVYGK7hLy9f7mj9bAqk&usqp=CAU" align="right"
     alt="Logo image detection" width="80" height="80">
     
Este é um projeto de machine learning que consiste na utilização de redes neurais convolucionais na 
determinação de idade por meio de fotos de rosto. O projeto foi desenvolvido em Python e conta com
bibliotecas como Tensorflow, Keras e OpenCV. Ele também é a parte principal do serviço (API) 
desenvolvido para extrair rostos em imagens utilizando o algoritmo de  Haar cascade do OpenCV.

* É posssível treinar novos modelos e adicionar os novos pesos da rede neural. 
* O dataset utilizado neste projeto é de domínio público e pode ser encontrado no Kaggle. 
* É aconselhável o uso de GPUs para o treinamento da rede (Treinamento com RTX 3060 estimado mais de 1 hora para 100 epochs) 


# Datasets 
* https://www.kaggle.com/competitions/cs599-assignment-2-age-estimation/data
* https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

# Bibliotecas 
- [![tensorflow](https://badges.aleen42.com/src/tensorflow.svg)]() - Tensorflow 2.6.0
- [![python](https://badges.aleen42.com/src/python.svg)]() - Python 3.7
- ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white) - CUDA Toolkit 

# Instalação Rápida

Caso prefira explorar a biblioteca, é possível realizar a instalação do projeto por meio do seguinte comando:
```
pip3 install git+https://github.com/vitorglemos/age-classification.git
```

Se preferir, é possível testar o projeto online por meio das instruções a seguir:

# Utilização (API)
