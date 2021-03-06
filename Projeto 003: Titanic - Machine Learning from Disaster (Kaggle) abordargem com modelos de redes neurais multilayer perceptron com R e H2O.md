## Projeto 3: Titanic - Machine Learning from Disaster (Kaggle) abordargem com modelos de Deep Learning com R e H2O.

O naufrágio do RMS Titanic é um dos mais infames naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios. Uma das razões pelas quais o naufrágio causou tal perda de vida foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta. Neste desafio, pedimos que você conclua a análise de quais tipos de pessoas provavelmente sobreviveriam. Para esse desafio irei abordar um modelo de classificação via redes neurais deep learning no h2o, pois estamos falando de um desenvolvimento de modelo não linear.

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
library(h2o);
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(rpart);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = read.csv2("treino.csv")
teste = read.csv2("teste.csv")
teste = cbind(teste,Survived = NA)
full = rbind(treino,teste)
dim(full)
```

### Análise explorátoria de dados.

* Análise dos dados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(full,Target="Survived",op_file = "EDA_titanic.html")
```

### Features Engineering.

* Titulo:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Name = as.character(full$Name)
full$Title = sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
full$Title = sub(' ', '', full$Title)
full$Title[full$Title %in% c('Mme', 'Mlle')] = 'Mlle'
full$Title[full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
full$Title[full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
full$Title = as.factor(full$Title)
```
* Tamanho da familia:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$FamilySize = full$SibSp + full$Parch + 1
full$FamilySize = as.factor(full$FamilySize)
```
* Local de embarque:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Embarked[c(62,830)] = "S"
full$Embarked = as.factor(full$Embarked)
```
* Preenchimento de valores faltantes na variável tarifa e padronização:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Fare[1039] = median(full$Fare, na.rm=TRUE)
full$Fare = log(full$Fare)
```
* Preenchimento de valores faltantes na variável idade e padronização:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
age_miss = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
               data=full[!is.na(full$Age),], method="anova")
               
full$Age[is.na(full$Age)] = predict(age_miss, full[is.na(full$Age),])
full$Age = log(full$Age)
```
* Criação da variavél categorica Pclass:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Pclass = as.factor(full$Pclass)
```
* Criação da variavél categorica Cabine:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
titanic$CabinType = substr(titanic$Cabin, 1, 1)
titanic$CabinType = as.factor(titanic$CabinType)
```
* Criação da variavél grupo crianças (Idade <18):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Child[full$Age < 18] = 'Child'
full$Child[full$Age >= 18] = 'Adult'
full$Child = as.factor(full$Child)
```
* Criação da variavél grupo tamanho da familia:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$FsizeD[full$FamilySize == 1] = 'sozinho'
full$FsizeD[full$FamilySize < 5 & full$FamilySize > 1] = 'pequena_familia'
full$FsizeD[full$FamilySize > 4] = 'grande_familia'
```
* Criação da variavél grupo tamanho da familia:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$mae = 0
full$mae[full$Sex=='female' & full$Parch>0 & full$Age>18 & full$Title!='Miss']< = 1
full$mae = as.factor(full$mae)
```
* Retirada de dados não modelaveis:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full = full %>% select(-PassengerId,-Name,-SibSp,-Parch,-Ticket)
```
* Criação de variaveis dummy:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dummy = dummyVars(" ~ .", data = full)
full = data.frame(predict(dummy, newdata = full))
View(full)
dim(full)
```
* Criação do target:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
full$Survived = as.factor(full$Survived)
```

### Divisão do dataset.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = full[1:891,]
teste = full[892:1309,]
```

### Análise explorátoria de dados do dataset.

* Análise dos dados com foco no target:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(treino,Target="Survived",op_file = "EDA_titanic_trans.html")
```
* Matriz de correlação:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(treino,label = T,nbreaks = 5,label_round = 2)
```
### Seleção de variáveis.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = treino %>% select(Pclass.1,Pclass.3,Sex.male,Sex.female,Title.Miss,Title.Mr,Title.Mrs,Embarked.C,Embarked.S,FamilySize.1,FamilySize.2,Survived)
teste = teste %>% select(Pclass.1,Pclass.3,Sex.male,Sex.female,Title.Miss,Title.Mr,Title.Mrs,Embarked.C,Embarked.S,FamilySize.1,FamilySize.2)
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

* Lista de hiperparametros:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
activation_opt = c("Rectifier", "RectifierWithDropout")
l1_opt = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
l2_opt = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1)
hyper_params = list(activation = activation_opt,l1 = l1_opt,l2 = l2_opt)
search_criteria = list(strategy = "Cartesian", max_runtime_secs = 5000)
```

* Grid Search Deep learning:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dl_grid = h2o.grid("deeplearning", x = 1:11, y = "Survived",
                    grid_id = "dl_grid",
                    training_frame = treino.hex,
                    seed = 74,
                    hidden = c(8,8,6,6,3),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)
```

* Resultados do Grid:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dl_gridperf = h2o.getGrid(grid_id = "dl_grid",sort_by = "MSE", decreasing = TRUE)
print(dl_gridperf)
```

* Selecionar o model_id para o modelo top DL, escolhido pela validação MSE:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
best_dl_model_id = dl_gridperf@model_ids[[1]]
best_dl = h2o.getModel(best_dl_model_id)
```

* Verficação da importância das variáveis do modelo:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
h2o.varimp_plot(best_dl)
```

* Predições:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
prediction = h2o.predict(best_dl, newdata = teste.hex)
```

### Solução ao Kaggle

```{r, cache=FALSE, message=FALSE, warning=FALSE}
solucao = data.frame(PassengerId = teste[1], Survived = prediction[1])
write.csv(solucao, file="solucao_deep_learning_h2o.csv",  row.names = FALSE)
```
### Resultado da submissão ao Kaggle

Modelando o dataset e utilizando o Grid e escolhendo o melhor modelo de deep learning de acordo com logloss conseguimos uma pontuação de 0.75119 e com isso na competição leva a posição 8597 de 10034 e assim para melhorar a acuracia podemos rodar o modelo com uma seleção de variaveis com base no iv.
