## Projeto 3: Titanic - Machine Learning from Disaster (Kaggle) abordargem com modelos de Deep Learning com R e H2O.

O naufrágio do RMS Titanic é um dos mais infames naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios. Uma das razões pelas quais o naufrágio causou tal perda de vida foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta. Neste desafio, pedimos que você conclua a análise de quais tipos de pessoas provavelmente sobreviveriam. Para esse desafio irei abordar um modelo de classificação via redes neurais multilayer perceptron no h2o.

https://www.kaggle.com/c/titanic/data

Informações sobre a variáveis:

Pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

Sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

Parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(h2o);library(dplyr);library(caret);
library(stringr);library(DMwR);library(Amelia);
library(SmartEDA);library(GGally);library(ggplot2);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = read.csv2("treino.csv")
teste = read.csv2("teste.csv")
full = rbind(treino,teste)
```

### Análise explorátoria de dados.

* Análise dos dados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(full,by="A",gp="Survived",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
ExpNumViz(full,gp="Survived",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```

### Criação de novas variáveis

* Titulo:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Name = as.character(full$Name)
full$Title = sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
full$Title = sub(' ', '', full$Title)
full$Title[full$Title %in% c('Mme', 'Mlle')] = 'Mlle'
full$Title[full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
full$Title[full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
full$Title = factor(full$Title)
```
* Sobrenome:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Surname = sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
```
* Tamanho da familia:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$FamilySize = full$SibSp + full$Parch + 1
```
* Local de embarque:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Embarked[c(62,830)] = "S"
full$Embarked = factor(full$Embarked)
```
* Preenchimento de valores faltantes na variável tarifa:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Fare[1044] = median(full$Fare, na.rm=TRUE)
```
* Preenchimento de valores faltantes na variável idade:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
Agefit = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,data=full[!is.na(full$Age),], method="anova")
full$Age[is.na(full$Age)] = predict(Agefit, full[is.na(full$Age),])
```

### Criação de variaveis dummy.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
dummy = dummyVars(" ~ .", data = treino)
treino = data.frame(predict(dummy, newdata = treino))
print(treino)
```

### Análise explorátoria de dados do dataset 

* Análise dos dados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(full,by="A",gp="Survived",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
ExpNumViz(full,gp="Survived",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```
* Matriz de correlação
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(full,label = T,nbreaks = 5,label_round = 4)
```

### Inicialização do H2O.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
h2o.init()
```

### Entrada de dados no H2O.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino.hex = as.h2o(treino, destination_frame="treino.hex")
teste.hex = as.h2o(teste, destination_frame="teste.hex")
```

### Grid Search, Seleção do modelo e Teste.

* Deep learning hiperparametros
```{r, cache=FALSE, message=FALSE, warning=FALSE}
activation_opt = c("Rectifier", "RectifierWithDropout", "Maxout", "MaxoutWithDropout")
l1_opt = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hyper_params = list(activation = activation_opt,l1 = l1_opt,l2 = l2_opt)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 120)
```

* Grid Search Deep learning
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dl_grid = h2o.grid("deeplearning", x = x, y = y,
                    grid_id = "dl_grid",
                    training_frame = treino,
                    validation_frame = teste,
                    seed = 1,
                    hidden = c(10,10),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
```

* Resultados do Grid
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dl_gridperf = h2o.getGrid(grid_id = "dl_grid",sort_by = "auc", decreasing = TRUE)
print(dl_gridperf)
```

* Selecionar o model_id para o modelo top DL, escolhido pela validação AUC
```{r, cache=FALSE, message=FALSE, warning=FALSE}
best_dl_model_id = dl_gridperf@model_ids[[1]]
best_dl = h2o.getModel(best_dl_model_id)
```

* Avaliação do desempenho do modelo em um conjunto de testes, para obtermos a performance do modelo escolhido
```{r, cache=FALSE, message=FALSE, warning=FALSE}
best_dl_perf = h2o.performance(model = best_dl,newdata = teste)
h2o.auc(best_dl_perf)
h2o.confusionMatrix(best_dl_perf, teste)
```
