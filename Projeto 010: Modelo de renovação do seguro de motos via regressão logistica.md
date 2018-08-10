# Projeto 10: Modelo de renovação do seguro de motos comparação entre o modelo de regressão logística e árvores de decisão CART (Problema de classes desbalanceadas).

## Pacotes.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(readr);
library(caret);
library(dplyr);
library(ROSE);
library(DMwR);
library(Amelia);
library(pROC);
library(doParallel);
```

## Entradas de dados.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
BASE_REGRAS_MOTO = read.csv("C:/Users/vd114342/Desktop/BASE_REGRAS_MOTO.csv", sep=";")
View(BASE_REGRAS_MOTO)
```

## Transformação de dados.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = BASE_REGRAS_MOTO
data$QTD_PROP = as.factor(data$QTD_PROP)
str(data)
```

## Verificação de dados faltantes.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
missmap(data)
data = na.omit(data)
```

## Seleção de variaveis.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data = data %>% select(COD_REGIAO_POLITICA,FAMILIA_GRUPO_AUTO2,var_reg_final_for,
                       FLG_PARCERIA_VALID,diferenca50_c,fat_tt_glm2,tx_comer_renov,
                       dsc_porte_final,QTD_PROP)
```

## Divisão do dataset.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(8678)
part = createDataPartition(y = data$QTD_PROP, p = 0.8, list = FALSE)
treino = data[part,]
teste = data[-part,]
```

## Analise da amostragem.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
# Analise das proporções da variavel target.
tab_data = round(prop.table(table(data$QTD_PROP)),2)
tab_treino = round(prop.table(table(treino$QTD_PROP)),2)
tab_teste = round(prop.table(table(teste$QTD_PROP)),2)
```

## Resumos das variaveis importantes entre as bases.

* Risco
```{r, cache=FALSE, message=FALSE, warning=FALSE}
summary(data$fat_tt_glm2)
summary(treino$fat_tt_glm2)
summary(teste$fat_tt_glm2)
par(mfrow=c(3,1))
hist(data$fat_tt_glm2)
hist(treino$fat_tt_glm2)
hist(teste$fat_tt_glm2)
```
## Balanceamento de base.

* Undersampling
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(9567)
under = downSample(x = (treino %>% select(-QTD_PROP)), y = treino$QTD_PROP)
tab_under = round(prop.table(table(under$Class)),2)
summary(under)
```
* Oversampling
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(8475)
over = upSample(x = (treino %>% select(-QTD_PROP)), y = treino$QTD_PROP)                         
tab_over = round(prop.table(table(over$Class)),2)
summary(over)
```
* Método ROSE
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(9560)
rose = ROSE(QTD_PROP ~ ., data  = treino)$data                         
tab_rose = round(prop.table(table(rose$QTD_PROP)),2)
summary(rose)
```
* Método SMOTE
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(9574)
smote = SMOTE(QTD_PROP ~ ., data  = treino)$data                         
tab_smote = round(prop.table(table(smote$QTD_PROP)),2)
summary(smote)
```
### Avaliação das variáveis na amostragens.

* Variavel de risco
```{r, cache=FALSE, message=FALSE, warning=FALSE}
t.test(data$fat_tt_glm2,under$fat_tt_glm2)
t.test(data$fat_tt_glm2,over$fat_tt_glm2)
t.test(data$fat_tt_glm2,rose$fat_tt_glm2)
t.test(data$fat_tt_glm2,smote$fat_tt_glm2)
```

### Preparação do treinamento.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
control = trainControl(method= "cv",number = 5, allowParallel = TRUE)
```

### Treinamento do modelo GLM.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(106)
modelglm = train(QTD_PROP~., data=treino, method="glm", trControl=control)
set.seed(197)
modelglm_under = train(Class~., data=under, method="glm", trControl=control)
set.seed(112)
modelglm_over = train(Class~., data=over, method="glm", trControl=control)
set.seed(788)
modelglm_rose = train(QTD_PROP~., data=rose, method="glm", trControl=control)
set.seed(788)
modelglm_smote = train(QTD_PROP~., data=smote, method="glm", trControl=control)
```

### Treinamento do modelo árvore de decisão CART.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(106)
modelrpart = train(QTD_PROP~., data=treino, method="rpart", trControl=control)
set.seed(197)
modelrpart_under = train(Class~., data=under, method="rpart", trControl=control)
set.seed(112)
modelrpart_over = train(Class~., data=over, method="rpart", trControl=control)
set.seed(788)
modelrpart_rose = train(QTD_PROP~., data=rose, method="rpart", trControl=control)
set.seed(781)
modelrpart_smote = train(QTD_PROP~., data=smote, method="rpart", trControl=control)
```

### Verificação dos resultados.

* Resultados GLM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados_glm = resamples(list(treino=modelglm,treino_under=modelglm_under,treino_over=modelglm_over,treino_rose=modelglm_rose, treino_smote=modelglm_smote))
bwplot(resultados_glm)
dotplot(resultados_glm)
```
* Resultados CART
```{r, cache=FALSE, message=FALSE, warning=FALSE}
resultados_rpart = resamples(list(treino=modelrpart,treino_under=modelrpart_under,treino_over=modelrpart_over,treino_rose=modelrpart_rose,treino_smote=modelrpart_smote))
bwplot(resultados_rpart)
dotplot(resultados_rpart)
```
### Testes e Estatísticas

* Predições GLM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_modelglm = predict(modelglm,teste)
pred_modelglm_under = predict(modelglm_under,teste)
pred_modelglm_over = predict(modelglm_over,teste)
pred_modelglm_rose = predict(modelglm_rose,teste)
pred_modelglm_smote = predict(modelglm_smote,teste)
```
* Matriz de confusão dos modelos GLM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
desbal_glm = confusionMatrix(pred_modelglm,teste$QTD_PROP)
under_glm = confusionMatrix(pred_modelglm_under,teste$QTD_PROP)
over_glm = confusionMatrix(pred_modelglm_over,teste$QTD_PROP)
rose_glm = confusionMatrix(pred_modelglm_rose,teste$QTD_PROP)
smote_glm = confusionMatrix(pred_modelglm_smote,teste$QTD_PROP)
```
* Tabela de métricas dos modelos GLM
```{r, cache=FALSE, message=FALSE, warning=FALSE}
desbal_estat_glm = desbal$byClass
under_estat_glm = under$byClass
over_estat_glm = over$byClass
rose_estat_glm = rose$byClass
smote_estat_glm = smote$byClass
tab_estat_glm = cbind(desbal_estat_glm,under_estat_glm,over_estat_glm,rose_estat_glm,smote_estat_glm)
```
* Predições AD CART
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_modelrpart = predict(modelrpart,teste)
pred_modelrpart_under = predict(modelrpart_under,teste)
pred_modelrpart_over = predict(modelrpart_over,teste)
pred_modelrpart_rose = predict(modelrpart_rose,teste)
pred_modelrpart_smote = predict(modelrpart_smote,teste)
```
* Matriz de confusão dos modelos AD CART
```{r, cache=FALSE, message=FALSE, warning=FALSE}
desbal_rpart = confusionMatrix(pred_modelrpart,teste$QTD_PROP)
under_rpart = confusionMatrix(pred_modelrpart_under,teste$QTD_PROP)
over_rpart = confusionMatrix(pred_modelrpart_over,teste$QTD_PROP)
rose_rpart = confusionMatrix(pred_modelrpart_rose,teste$QTD_PROP)
smote_rpart = confusionMatrix(pred_modelrpart_smote,teste$QTD_PROP)
```
* Tabela de métricas dos modelos AD CART
```{r, cache=FALSE, message=FALSE, warning=FALSE}
desbal_estat_rpart = desbal$byClass
under_estat_rpart = under$byClass
over_estat_rpart = over$byClass
rose_estat_rpart = rose$byClass
smote_estat_rpart = smote$byClass
tab_estat_rpart = cbind(desbal_estat_rpart,under_estat_rpart,over_estat_rpart,rose_estat_rpart,smote_estat_rpart)
```
### Conclusões
