# Projeto 4: Criação de um modelo de classificação Xgboost sobre a frequencia de doadores de sangue na cidade de Hsin-Chu, em Taiwan

Este estudo adotou o banco de dados de doadores do Centro de Serviços de Transfusão de Sangue na cidade de Hsin-Chu, em Taiwan. O centro passa seu ônibus de serviço de transfusão de sangue para uma universidade na cidade de Hsin-Chu para coletar sangue doado a cada três meses. Para construir um modelo de machine learning, selecionamos aleatoriamente 748 doadores do banco de dados do doador. Esses 748 dados de doadores, cada um incluindo R (Recência - meses desde a última doação), F (Frequência - número total de doações), M (monetária - total de sangue doado em cc), T (tempo - meses desde a primeira doação) e uma variável binária representando se doou sangue em março de 2007 (1 representa doar sangue; 0 significa não doar sangue). De acordo com uma análise explorátoria de dados e um entendimento dos dados podemos abordar um modelo de classificação com base numa árvore de decisão e tambem num modelo Xgboost.

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr);
library(caret);
library(ROSE);
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
datadb = predict(pp_data, newdata = datadb)
head(datadb)
```
* Divisão do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(86)
part = createDataPartition(y = datadb$class, p = 0.8, list = FALSE)
treino = datadb[part,]
teste = datadb[-part,]
```
* Controle do treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
control = trainControl(method = "cv",number = 10,allowParallel = TRUE)
```

### Balanceamento de classes.

* Undersampling
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(9567)
down_train = downSample(x = treino[,-5], y = treino$class)
table(down_train$Class)
```
* Oversampling
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(8475)
up_train = upSample(x = treino[,-5], y = treino$class)                         
table(up_train$Class)
```
* Método ROSE
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(9560)
rose_train = ROSE(class ~ ., data  = treino)$data                         
table(rose_train$class)
```

### Seleção do modelo.

* Modelo Xgboost (Undersampling)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(1784)
modelxgbTree_u = train(Class~., data=down_train, method="xgbTree", trControl=control)
```
* Modelo Xgboost (Oversampling)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(1784)
modelxgbTree_o = train(Class~., data=up_train, method="xgbTree", trControl=control)
```
* Modelo Xgboost (ROSE)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(1787)
modelxgbTree_r = train(class~., data=rose_train, method="xgbTree", trControl=control)
```
* Resultados do treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados = resamples(list(treino_over=modelxgbTree_o, treino_under=modelxgbTree_u, treino_rose=modelxgbTree_r))
bwplot(resultados)
dotplot(resultados)
```

### Tuning.

* Grid
```{r, cache=FALSE, message=FALSE, warning=FALSE}
grid = expand.grid(nrounds = c(10,50,100,500),
                   eta = c(0.1, 0.01, 0.001, 0.0001),
                   max_depth = c(2, 4, 6, 8, 10),
                   gamma = 1,
                   colsample_bytree = 1,
                   min_child_weight = 100,
                   subsample = 1)
```
* Treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgboost.tune = train(x=(up_train %>% select(-Class)),
                     y=as.factor(up_train$Class),
                     method = "xgbTree",
                     trControl = control,
                     tuneGrid=grid,
                     verbose=FALSE)
```
* Resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgboost.tune$bestTune
plot(xgboost.tune)  
res = xgboost.tune$results
```

### Teste e avaliação de perfomance.

* Predições
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgb.pred = predict(xgb.tune,teste)
```
* Matriz de confusão e métricas relacionadas ao teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
confusionMatrix(xgb.pred,teste$class,positive = "1")
```
* Obtenção das probabilidades
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgb.probs = predict(xgb.tune,teste,type="prob")
head(gbm.probs)
```
* Curva ROC
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgb.ROC = roc(predictor=xgb.probs,response=teste$class,levels=rev(levels(teste$class)))
plot(xgb.ROC,main="Curva ROC da frequencia de doadores de sangue")
```
* Área da curva ROC
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgb.ROC$auc
```

### Salvar o modelo treinado e testado.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
saveRDS(xgb.tune,file = "modelo_doadores_sangue")
```
