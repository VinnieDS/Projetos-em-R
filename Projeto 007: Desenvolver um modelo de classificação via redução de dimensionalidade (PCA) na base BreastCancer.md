# Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base BreastCancer.

O objetivo é identificar cada uma das várias classes benignas ou malignas. As amostras chegam periodicamente como o Dr. Wolberg relata seus casos clínicos. O banco de dados, portanto, reflete essa ordem cronológica agrupamento dos dados. Esta informação de agrupamento aparece imediatamente abaixo, tendo sido removida dos dados em si. Cada variável, exceto a primeira, foi convertida em 11 números numéricos primitivos. atributos com valores que variam de 0 a 10. Há 16 valores de atributos ausentes. Uma base de dados com 699 observações em 11 variáveis, sendo uma delas uma variável de caráter, 9 sendo ordenada ou nominal e 1 classe alvo.

1] Id - Sample code number

2] Cl.thickness - Clump Thickness

3] Cell.size - Uniformity of Cell Size

4] Cell.shape - Uniformity of Cell Shape

5] Marg.adhesion - Marginal Adhesion

6] Epith.c.size - Single Epithelial Cell Size

7] Bare.nuclei - Bare Nuclei

8] Bl.cromatin - Bland Chromatin

9] Normal.nucleoli - Normal Nucleoli

10] Mitoses - Mitoses

11] Class - Class

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(rpart);
library(e1079);
library(Amelia);
library(mlbench);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(BreastCancer)
dim(BreastCancer)
```

### Análise explorátoria de dados.

* Análise dos dados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(BreastCancer,Target="Class",op_file = "EDA_BreastCancer.html")
```

### Features Engineering.

* Transformação dos fatores em dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
BreastCancer$Cl.thickness = as.numeric(BreastCancer$Cl.thickness)
BreastCancer$Cell.size = as.numeric(BreastCancer$Cell.size)
BreastCancer$Cell.shape = as.numeric(BreastCancer$Cell.shape)
BreastCancer$Marg.adhesion = as.numeric(BreastCancer$Marg.adhesion)
BreastCancer$Epith.c.size = as.numeric(BreastCancer$Epith.c.size)
BreastCancer$Bare.nuclei = as.numeric(BreastCancer$Bare.nuclei)
BreastCancer$Bl.cromatin = as.numeric(BreastCancer$Bl.cromatin)
BreastCancer$Normal.nucleoli = as.numeric(BreastCancer$Normal.nucleoli)
BreastCancer$Mitoses = as.numeric(BreastCancer$Mitoses)
```
* Mudança do nome para modelagem
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = BreastCancer
```

* Verificação dos dados faltantes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
bare_nuclei_miss = rpart(Bare.nuclei ~ Cl.thickness + Cell.size + Cell.shape + Marg.adhesion + Epith.c.size + 
                         Epith.c.size + Bl.cromatin + Normal.nucleoli, 
                         data=data[!is.na(data$Bare.nuclei),], method="anova")
              
data$Bare.nuclei[is.na(data$Bare.nuclei)] = predict(bare_nuclei_miss, data[is.na(data$Bare.nuclei),])
```
### Análise explorátoria de dados.

* Análise de dados modelados e com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(data,Target="Class",op_file = "EDA_BreastCancer_trans.html")
```
* Matriz de correlação
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorrplot(data,label = T,nbreaks = 5,label_round = 2)
```

### Seleção de variáveis

```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = data %>% select()
```

### Preparação para o treinamento.

* Divisão do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(86)
part = createDataPartition(y = data$Class, p = 0.8, list = FALSE)
treino = data[part,]
teste = data[-part,]
```

* Pré processamento com PCA
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pp_data = preProcess(data[, -8], method = c("pca"))
data = predict(pp_data, newdata = data[, -8])
head(data)
```

* Controle do treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
control = trainControl(method = "cv",number = 5,allowParallel = TRUE)
```

### Seleção de modelo

* Modelos de Naive Bayes, Linear Generalizado, Gradient Boosted, KNN e Random Forest
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(95)
modelnb = train(Class~., data=treino, method="nb", trControl=control)
set.seed(71)
modelglm = train(Class~., data=treino, method="glm", trControl=control)
set.seed(78)
modelknn = train(Class~., data=treino, method="knn", trControl=control)
set.seed(80)
modelgbm = train(Class~., data=treino, method="gbm", trControl=control)
set.seed(97)
modelrf = train(Class~., data=treino, method="rf", trControl=control)
```

* Agregação dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados = resamples(list(NB=modelLnb, GLM=modelglm, KNN=modelknn, GBM=modelGbm, Rf=modelrf))
```

* Resumo dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
summary(resultados)
```

* Boxplots dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
bwplot(resultados)
```

* Dot plots dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dotplot(resultados)
```

### Tuning 

* O melhor modelo de acordo com os resultados acima é o gbm
```{r, cache=FALSE, message=FALSE, warning=FALSE}
grid = expand.grid(interaction.depth=c(1,2), 
                    n.trees=c(10,20),
                    shrinkage=c(0.01,0.1),
                    n.minobsinnode = 20)      

gbm.tune = train(x=treino[1:10],y=treino$Class,
                              method = "gbm",
                              metric = "ROC",
                              trControl = ctrl,
                              tuneGrid=grid,
                              verbose=FALSE)
gbm.tune$bestTune
plot(gbm.tune)  
res = gbm.tune$results
```

### Teste e avaliação de perfomance
