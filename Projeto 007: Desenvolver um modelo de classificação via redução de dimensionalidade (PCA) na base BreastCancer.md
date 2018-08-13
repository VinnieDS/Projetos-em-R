# Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base Breast Cancer.

O objetivo é identificar cada uma das várias classes benignas ou malignas. As amostras chegam periodicamente como o Dr. Wolberg relata seus casos clínicos. O banco de dados, portanto, reflete essa ordem cronológica agrupamento dos dados. Esta informação de agrupamento aparece imediatamente abaixo, tendo sido removida dos dados em si. Cada variável, exceto a primeira, foi convertida em 11 números numéricos primitivos. atributos com valores que variam de 0 a 10. Há 16 valores de atributos ausentes. Uma base de dados com 699 observações em 11 variáveis, sendo uma delas uma variável de caráter, 9 sendo ordenada ou nominal e 1 classe alvo. Iremos realizar uma redução de dimensionalidade em vez da seleção de variáveis, pois os preditores desse dataset são altamente correlacionados e isso pode atrapalhar nas predições.

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
library(devtools);
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(rpart);
library(shiny);
library(Amelia);
library(pROC);
library(gbm);
library(mlbench);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(BreastCancer)
dim(BreastCancer)
```

### Análise explorátoria de dados.

* Análise dos dados brutos com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(BreastCancer,Target="Class",op_file = "EDA_BreastCancer.html")
```

### Features Engineering.

* Mudança do nome para modelagem e retirada do ID
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = BreastCancer
data = data[,-1]
dim(data)
```
* Transformação dos fatores em dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data_if = data[,-10] %>% mutate_if(is.factor, as.numeric)
data = cbind(data_if,data[10])
str(data)
```
* Verificação dos dados faltantes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
missmap(data)
```
* Preenchimento dos dados faltantes via algoritmo rpart (Variável "Bare.nuclei")
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
* Matriz de correlação (dados altamente correlacionados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(data[-10],label = T,nbreaks = 5,label_round = 2)
```

### Analise de componentes principais.

* Desenvolvimento da analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pc = prcomp(data[,-10], center = TRUE, scale. = TRUE)
print(pc)
summary(pc)
```

* Gráfico da analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
install_github("ggbiplot", "vqv")
library(ggbiplot)
g = ggbiplot(pc, obs.scale = 1, var.scale = 1, 
             groups = data$Class, ellipse = TRUE, 
             circle = TRUE, ellipse.prob = 0.68)
g = g+scale_alpha_discrete(name = '')
g = g+theme(legend.direction = 'horizontal', legend.position = 'top')
print(g)
```

### Preparação para o treinamento.

* Pré processamento com analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pp_data = preProcess(data[, -10], method = c("pca"))
data = predict(pp_data, newdata = data[, -11])
head(data)
```
* Matriz de correlação com o dado pre processado pela analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(data[2:8],label = T,nbreaks = 5,label_round = 2)
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

### Seleção de modelo.

* Modelos de Naive Bayes, C50, GLM, GBM, KNN, Random Forest, Xgboost e SVM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(95)
modelnb = train(Class~., data=treino, method="nb", trControl=control)
set.seed(40)
modelc50 = train(Class~., data=treino, method="C5.0", trControl=control)
set.seed(71)
modelglm = train(Class~., data=treino, method="glm", trControl=control)
set.seed(78)
modelknn = train(Class~., data=treino, method="knn", trControl=control)
set.seed(80)
modelgbm = train(Class~., data=treino, method="gbm", trControl=control)
set.seed(97)
modelrf = train(Class~., data=treino, method="rf", trControl=control)
set.seed(11)
modelxgbTree = train(Class~., data=treino, method="xgbTree", trControl=control)
set.seed(75)
modelsvmRadial = train(Class~., data=treino, method="svmRadial", trControl=control)
```

* Agregação dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados = resamples(list(NB=modelnb, C50=modelc50, GLM=modelglm, 
                            KNN=modelknn, GBM=modelgbm, RF=modelrf, 
                            XGB=modelxgbTree, SVM=modelsvmRadial))
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

* Comparação estatística dos resultados (Entre os melhores modelos)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
compare_models(modelc50, modelgbm)
```

* Conclusão

De acordo com a acuracia vamos utilizar o modelo gbm para modelar os dados.

### Tuning - Melhorar a performance e evitar o overfiting.

* Grid 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
grid = expand.grid(interaction.depth=c(1,2), 
                    n.trees=c(10,20,30,40,50,100),
                    shrinkage=c(0.001,0.01,0.1),
                    n.minobsinnode = 20)      
```
* Treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.tune = train(x=treino[2:8],y=treino$Class,
                              method = "gbm",
                              metric = "ROC",
                              trControl = control,
                              tuneGrid=grid,
                              verbose=FALSE)
```
* Resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.tune$bestTune
plot(gbm.tune)  
res = gbm.tune$results
```

### Teste e avaliação de perfomance.

* Predições
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.pred = predict(gbm.tune,teste)
```
* Matriz de confusão e métricas relacionadas ao teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
confusionMatrix(gbm.pred,teste$Class)
```
* Obtenção das probabilidades
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.probs = predict(gbm.tune,teste,type="prob")
head(gbm.probs)
```
* Curva ROC 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.ROC = roc(predictor=gbm.probs$benign,
               response=teste$Class,
               levels=rev(levels(teste$Class)))

plot(gbm.ROC,main="Diagnóstico de Cancer de Mama via GBM")
```
* Área da curva ROC
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.ROC$auc
```

### Salvar o modelo treinado e testado.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
saveRDS(gbm.tune,file = "modelo_cancer_final")
```

### Conclusão.

Utilizando a redução de dimensionalidade PCA conseguimos um modelo com auc de 0.9979 (classificador quase perfeito) e uma acuracia de 0.9784 nos dados de teste uma performance muito melhor do que a referencia que treinou os mesmos dados com o algoritmo de KNN (K=1) e obteve a acuracia de 0.95 e assim reduzindo a taxa de falsos positivos e falsos negativos. 

### Deployment in Shiny R (Criação do APP).

Utilizando o modelo desenvolvido que melhor explica a relação do cancer de mama do tipo benigno com o maligino iremos desenvolver um app no qual o usuario vai inserir os fatores determinantes para o tipo de cancer e o app vai retornar com o tipo e a probabilidade do diagnostico.
