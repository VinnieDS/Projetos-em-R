# Projeto 4: Criação de um modelo de classificação sobre a frequencia de doadores de sangue na cidade de Hsin-Chu, em Taiwan

Este estudo adotou o banco de dados de doadores do Centro de Serviços de Transfusão de Sangue na cidade de Hsin-Chu, em Taiwan. O centro passa seu ônibus de serviço de transfusão de sangue para uma universidade na cidade de Hsin-Chu para coletar sangue doado a cada três meses. Para construir um modelo de machine learning, selecionamos aleatoriamente 748 doadores do banco de dados do doador. Esses 748 dados de doadores, cada um incluindo R (Recência - meses desde a última doação), F (Frequência - número total de doações), M (monetária - total de sangue doado em cc), T (tempo - meses desde a primeira doação) e uma variável binária representando se doou sangue em março de 2007 (1 representa doar sangue; 0 significa não doar sangue). De acordo com uma análise explorátoria de dados e um entendimento dos dados podemos abordar um modelo de classificação com base numa árvore de decisão e tambem num modelo Xgboost.

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(Matrix);
library(Xgboost);
library(rpart);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
datadb = read.csv2("datadb.csv")
dim(datadb)
```

### Análise explorátoria de dados.

* Análise dos dados brutos com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(datadb,Target="class",op_file = "EDA_csts.html")
```

### Features Engineering.

* Transformação dos dados númericos em fatores
```{r, cache=FALSE, message=FALSE, warning=FALSE}
datadb$class = as.factor(datadb$class)
```

### Análise explorátoria de dados.

* Análise dos dados modelados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(datadb,Target="class",op_file = "EDA_csts.html")
```
* Matriz de correlação com o dado pre processado
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(datadb[1:4],label = T,nbreaks = 5,label_round = 2)
```

### Preparação para o treinamento.

* Pré-processamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pp_data = preProcess(datadb[1:4], method = c("scale"))
datadb = predict(pp_data, newdata = data[,1:4])
head(datadb)
```
* Divisão do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(86)
part = createDataPartition(y = data$Class, p = 0.8, list = FALSE)
treino = data[part,]
teste = data[-part,]
```
* Controle do treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
control = trainControl(method = "cv",number = 10,classProbs = TRUE,allowParallel = TRUE)
```

### Treino do modelo

* Modelo Xgboost
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(1784)
modelxgbTree = train(Class~., data=treino, method="xgbTree", trControl=control)
```
