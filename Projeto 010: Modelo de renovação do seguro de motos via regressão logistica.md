# Projeto 10: Modelo de renovação do seguro de motos via regressão logistica (Problema de classes desbalanceadas)

## Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(readr);
library(caret);
library(dplyr);
library(ROSE);
library(Amelia);
library(doParallel);
```

## Entradas de dados.

BASE_REGRAS_MOTO = read.csv("C:/Users/vd114342/Desktop/BASE_REGRAS_MOTO.csv", sep=";")
View(BASE_REGRAS_MOTO)

## Transformação de dados.

data = BASE_REGRAS_MOTO
data$QTD_PROP = as.factor(data$QTD_PROP)
str(data)

## Verificação de dados faltantes.

missmap(data)
data = na.omit(data)

## Seleção de variaveis

data = data %>% select(COD_REGIAO_POLITICA,FAMILIA_GRUPO_AUTO2,var_reg_final_for,FLG_PARCERIA_VALID,diferenca50_c,fat_tt_glm2,tx_comer_renov,dsc_porte_final,QTD_PROP)

## Divisão do dataset.
set.seed(8678)
part = createDataPartition(y = data$QTD_PROP, p = 0.8, list = FALSE)
treino = data[part,]
teste = data[-part,]

## Analise da amostragem.

# Analise das proporções da variavel target.
tab_data = round(prop.table(table(data$QTD_PROP)),2)
tab_treino = round(prop.table(table(treino$QTD_PROP)),2)
tab_teste = round(prop.table(table(teste$QTD_PROP)),2)

## Resumos das variaveis importantes entre as bases.

# Risco
summary(data$fat_tt_glm2)
summary(treino$fat_tt_glm2)
summary(teste$fat_tt_glm2)

par(mfrow=c(3,1))
hist(data$fat_tt_glm2)
hist(treino$fat_tt_glm2)
hist(teste$fat_tt_glm2)

## Balanceamento de base.

# Undersampling
set.seed(9567)
under = downSample(x = (treino %>% select(-QTD_PROP)), y = treino$QTD_PROP)
tab_under = round(prop.table(table(under$QTD_PROP)),2)
summary(under)

# Oversampling
set.seed(8475)
over = upSample(x = (treino %>% select(-QTD_PROP)), y = treino$QTD_PROP)                         
tab_over = round(prop.table(table(over$QTD_PROP)),2)
summary(over)

# Método ROSE
set.seed(9560)
rose = ROSE(QTD_PROP ~ ., data  = treino)$data                         
tab_rose = round(prop.table(table(rose$QTD_PROP)),2)
summary(rose)

# Avaliação das variáveis na amostragens.

t.test(data$fat_tt_glm2,under$fat_tt_glm2)
t.test(data$fat_tt_glm2,over$fat_tt_glm2)
t.test(data$fat_tt_glm2,rose$fat_tt_glm2)

# Preparação do treinamento.
control = trainControl(method= "cv",number = 3, allowParallel = TRUE)

# Treinamento do modelo glm sem balanceamento.
set.seed(106)
modelglm = train(QTD_PROP~., data=treino, method="glm", trControl=control)
set.seed(197)
modelglm_under = train(Class~., data=under, method="glm", trControl=control)
set.seed(112)
modelglm_over = train(Class~., data=over, method="glm", trControl=control)
set.seed(788)
modelglm_rose = train(QTD_PROP~., data=rose, method="glm", trControl=control)

# Treinamento do modelo árvore de decisão CART sem balanceamento.
set.seed(106)
modelrpart = train(QTD_PROP~., data=treino, method="rpart", trControl=control)
set.seed(197)
modelrpart_under = train(Class~., data=under, method="rpart", trControl=control)
set.seed(112)
modelrpart_over = train(Class~., data=over, method="rpart", trControl=control)
set.seed(788)
modelrpart_rose = train(QTD_PROP~., data=rose, method="rpart", trControl=control)

# Treinamento do modelo de KNN sem balanceamento.
set.seed(106)
modelknn = train(QTD_PROP~., data=treino, method="knn",trControl=control)
set.seed(197)
modelknn_under = train(Class~., data=under, method="knn",trControl=control)
set.seed(112)
modelknn_over = train(Class~., data=over, method="knn",trControl=control)
set.seed(788)
modelknn_rose = train(QTD_PROP~., data=rose, method="knn",trControl=control)

# Verificação dos resultados.

resultados_glm = resamples(list(treino=modelglm,treino_under=modelglm_under,treino_over=modelglm_over,treino_rose=modelglm_rose))
bwplot(resultados_glm)
dotplot(resultados_glm)

resultados_rpart = resamples(list(treino=modelrpart,treino_under=modelrpart_under,treino_over=modelrpart_over,treino_rose=modelrpart_rose))
bwplot(resultados_rpart)
dotplot(resultados_rpart)

resultados_knn = resamples(list(treino=modelknn,treino_under=modelknn_under,treino_over=modelknn_over,treino_rose=modelknn_rose))
bwplot(resultados_knn)
dotplot(resultados_knn)

## Testes.

# Predições GLM.

pred_modelglm = predict(modelglm,teste)
pred_modelglm_under = predict(modelglm_under,teste)
pred_modelglm_over = predict(modelglm_over,teste)
pred_modelglm_rose = predict(modelglm_rose,teste)

# Matriz de confusão GLM.

confusionMatrix(pred_modelglm,teste$QTD_PROP)
confusionMatrix(pred_modelglm_under,teste$QTD_PROP)
confusionMatrix(pred_modelglm_over,teste$QTD_PROP)
confusionMatrix(pred_modelglm_rose,teste$QTD_PROP)

# Predições AD CART.

pred_modelrpart = predict(modelrpart,teste)
pred_modelrpart_under = predict(modelrpart_under,teste)
pred_modelrpart_over = predict(modelrpart_over,teste)
pred_modelrpart_rose = predict(modelrpart_rose,teste)

# Matriz de confusão AD CART.

confusionMatrix(pred_modelrpart,teste$QTD_PROP)
confusionMatrix(pred_modelrpart_under,teste$QTD_PROP)
confusionMatrix(pred_modelrpart_over,teste$QTD_PROP)
confusionMatrix(pred_modelrpart_rose,teste$QTD_PROP)

# Predições KNN.

pred_modelknn = predict(modelknn,teste)
pred_modelknn_under = predict(modelknn_under,teste)
pred_modelknn_over = predict(modelknn_over,teste)
pred_modelknn_rose = predict(modelknn_rose,teste)

# Matriz de confusão KNN.

confusionMatrix(pred_modelknn,teste$QTD_PROP)
confusionMatrix(pred_modelknn_under,teste$QTD_PROP)
confusionMatrix(pred_modelknn_over,teste$QTD_PROP)
confusionMatrix(pred_modelknn_rose,teste$QTD_PROP)

## Gráficos da curva ROC.

# Performance Desbalanceado

curva_roc_glm = roc(predictor=pred_modelglm,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_rpart = roc(predictor=pred_modelrpart,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_knn = roc(predictor=pred_modelknn,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))

par(mfrow=c(3,1))
plot(curva_roc_glm,main="Modelo GLM Desbalanceado")
plot(curva_roc_rpart,main="Modelo CART Desbalanceado")
plot(curva_roc_knn,main="Modelo KNN Desbalanceado")

# Performance Undersampling

curva_roc_glm_under = roc(predictor=pred_modelglm_under,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_rpart_under = roc(predictor=pred_modelrpart_under,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_knn_under = roc(predictor=pred_modelknn_under,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))

par(mfrow=c(3,1))
plot(curva_roc_glm_under,main="Modelo GLM Undersampling")
plot(curva_roc_rpart_under,main="Modelo CART Undersampling")
plot(curva_roc_knn_under,main="Modelo KNN Undersampling")

# Performance Oversampling

curva_roc_glm_over = roc(predictor=pred_modelglm_over,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_rpart_over = roc(predictor=pred_modelrpart_over,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_knn_over = roc(predictor=pred_modelknn_over,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))

par(mfrow=c(3,1))
plot(curva_roc_glm_over,main="Modelo GLM Oversampling")
plot(curva_roc_rpart_over,main="Modelo CART Oversampling")
plot(curva_roc_knn_over,main="Modelo KNN Oversampling")

# Performance ROSE sampling

curva_roc_glm_rose = roc(predictor=pred_modelglm_rose,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_rpart_rose = roc(predictor=pred_modelrpart_rose,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))
curva_roc_knn_rose = roc(predictor=pred_modelknn_rose,response=teste$QTD_PROP,levels=rev(levels(teste$QTD_PROP)))

par(mfrow=c(3,1))
plot(curva_roc_glm_rose,main="Modelo GLM Rose sampling")
plot(curva_roc_rpart_rose,main="Modelo CART Rose sampling")
plot(curva_roc_knn_rose,main="Modelo KNN Rose sampling")
