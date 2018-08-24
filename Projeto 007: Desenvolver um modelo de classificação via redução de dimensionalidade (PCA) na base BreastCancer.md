# Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base Breast Cancer.

As características são calculadas a partir de uma imagem digitalizada de um aspirador de agulha fina (PAAF) de uma massa mamária. Eles descrevem características dos núcleos celulares presentes na imagem. O espaço tridimensional é o descrito em: [KP Bennett e OL Mangasarian: "Discriminação Linear de Programação Robusta de Dois Conjuntos Linearmente Inseparáveis", Optimization Methods and Software 1, 1992, 23-34]. Esta base de dados também está disponível através do servidor ftp da UW CS: ftp ftp.cs.wisc.edu cd math-prog / cpo-dataset / machine-learn / WDBC / Também pode ser encontrado no UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Informações sobre Atributos:

1) Número ID 

2) Diagnóstico (M = maligno, B = benigno) 

Dez características reais são calculadas para cada núcleo celular:

a) raio (média das distâncias do centro para os pontos no perímetro) 

b) textura (desvio padrão dos valores da escala de cinza) 

c) perímetro 

d) área 

e) suavidade (variação local no comprimento do raio) 

f) compactação (perímetro ^ 2 / área - 1.0) 

g) concavidade (gravidade das porções côncavas do contorno) 

h) pontos côncavos (número de porções côncavas do contorno)

i) simetria 

j) dimensão fractal ("aproximação costeira" - 1)

A média, erro padrão e "pior" ou maior (média dos três maiores valores) desses recursos foram calculados para cada imagem, resultando em 30 recursos. Por exemplo, o campo 3 é o raio médio, o campo 13 é o raio SE, o campo 23 é o pior raio.

Todos os valores de recursos são recodificados com quatro dígitos significativos.
Valores de atributo ausentes: nenhum
Distribuição de classes: 357 benignas, 212 malignas

https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/home

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr)
library(caret)
library(ggplot2)
library(GGally)
library(Amelia)
library(pROC)
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = read.csv("../input/data.csv" , header = TRUE , sep = ",")
dim(data)
str(data)
data = data[2:32]
str(data)
```

### Verificação de dados faltantes.

* Mapa de dados faltantes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
missmap(data, main = "Verificação de valores faltantes")
```

### Análise explorátoria de dados.

* Análise de dados modelados e com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggplot(data, aes(x=diagnosis))+geom_bar(stat="count", width=0.5, fill="steelblue")+theme_minimal()+ggtitle("Target")
```
* Matriz de correlação (dados altamente correlacionados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(data[-1],label = F,nbreaks = 5,hjust = 0.5, size = 3, color = "grey50", layout.exp = 2)
```

### Preparação para o treinamento.

* Pré processamento com analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pp_data = preProcess(data[, -1], method = c("pca"))
data = predict(pp_data, newdata = data)
head(data)
```
* Matriz de correlação com o dado pre processado pela analise de componentes principais
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(data[-1],label = T,nbreaks = 5,label_round = 2,label_size = 4,hjust = 0.5, size = 4, 
       color = "grey50", layout.exp = 2)
```
* Divisão do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(86)
part = createDataPartition(y = data$diagnosis, p = 0.9, list = FALSE)
treino = data[part,]
teste = data[-part,]
```
* Controle do treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
control = trainControl(method = "cv",number = 5,classProbs = TRUE)
```

### Seleção de modelo.

* Modelos de Naive Bayes, C50, GLM, GBM, KNN, Random Forest, Xgboost e SVM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(95)
modelsvm = train(diagnosis~., data=treino, method="svmRadial", trControl=control)
set.seed(80)
modelgbm = train(diagnosis~., data=treino, method="gbm", trControl=control)
```

* Agregação dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados = resamples(list(SVM = modelsvm,GBM = modelgbm))
```

* Resumo dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
summary(resultados)
```

* Boxplots dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
bwplot(resultados, main = "Seleção de Modelos")
```

* Dot plots dos resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dotplot(resultados, main = "Seleção de Modelos")
```

* Comparação estatística dos resultados (Entre os melhores modelos)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
compare_models(modelsvm, modelgbm)
```

* Conclusão

De acordo com a acuracia vamos utilizar o modelo gbm para modelar os dados.

### Tuning - Melhorar a performance e evitar o overfiting.

* Grid 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
grid = expand.grid(n.trees=c(20,30,50,100),
                   interaction.depth=c(1,2),
                   shrinkage=c(0.001,0.01,0.1),
                   n.minobsinnode = c(20,30))    
```
* Treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.tune = train(x=treino[2:11],y=treino$diagnosis,
                              method = "gbm",
                              metric = "ROC",
                              trControl = control,
                              tuneGrid=grid,
                              verbose=FALSE)
```
* Resultados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
plot(gbm.tune, main = "Tunning GBM")   
```

### Teste e avaliação de perfomance.

* Predições
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.pred = predict(gbm.tune,teste)
```
* Matriz de confusão e métricas relacionadas ao teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
confusionMatrix(gbm.pred,teste$diagnosis,positive = "B")
fourfoldplot(confusionMatrix(gbm.pred,teste$diagnosis,positive = "B")$table, main = "Matriz de confusão GBM Tune")
```
* Obtenção das probabilidades
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.probs = predict(gbm.tune,teste,type="prob")
head(gbm.probs)
```
* Curva ROC 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
gbm.ROC = roc(predictor=gbm.probs$B,response=teste$diagnosis)
plot(gbm.ROC,main="Diagnóstico de Cancer de Mama via GBM",
     print.auc=TRUE,auc.polygon=TRUE,auc.polygon.col="gray",print.thres=TRUE)
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

Utilizando a redução de dimensionalidade PCA conseguimos um modelo com auc de 0.9905 (classificador quase perfeito) e uma acuracia de 0.96 nos dados de teste uma performance muito melhor do que a referencia que treinou os mesmos dados com o algoritmo de KNN (K=1) e obteve a acuracia de 0.95 e assim reduzindo a taxa de falsos positivos e falsos negativos. 
