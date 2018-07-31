## Projeto 3: Titanic - Machine Learning from Disaster (Kaggle) abordargem com modelos de redes neurais multilayer perceptron com R e H2O.

O naufrágio do RMS Titanic é um dos mais infames naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios. Uma das razões pelas quais o naufrágio causou tal perda de vida foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta. Neste desafio, pedimos que você conclua a análise de quais tipos de pessoas provavelmente sobreviveriam. Para esse desafio irei abordar um modelo de classificação via redes neurais multilayer perceptron no h2o.

https://www.kaggle.com/c/titanic/data

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(h2o);library(dplyr);library(ggplot2);library(caret);library(stringr);
```

### Entrada de dados.

Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = read.csv2("treino.csv")
teste = read.csv2("teste.csv")
dim(treino)
dim(teste)
str(treino)
str(teste)
```

### Transformação dos dados.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino$Survived = as.factor(treino$Survived)
treino$Pclass = as.factor(treino$Pclass)
```

### Tratamento do dados faltantes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
missmap(treino)


```

