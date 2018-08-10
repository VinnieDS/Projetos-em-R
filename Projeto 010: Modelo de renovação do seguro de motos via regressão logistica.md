# Projeto 10: Modelo de renovação do seguro de motos via regressão logistica (Problema de classes desbalanceadas)

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

