## Projeto 1: Modelo de regressão para estimar o consumo de combustivel na base mtcars

Este conjunto de dados é uma versão ligeiramente modificada do conjunto de dados fornecido na biblioteca do StatLib. Em consonância com o uso de Ross Quinlan (1993) na previsão do atributo "mpg", 8 das instâncias originais foram removidas porque tinham valores desconhecidos para o atributo "mpg". O conjunto de dados original está disponível no arquivo "auto-mpg.data-original".

"Os dados dizem respeito ao consumo de combustível do ciclo urbano em milhas por galão, a ser previsto em termos de 3 atributos discretos e 5 contínuos de valor múltiplo." (Quinlan, 1993)

As tarefas são verificar na base de dados apresentado é possivel prever o consumo de combustível de cada carro (mpg) via modelo de regressão linear comparando o método de reamostragem de validação cruzada (10) com a validação holdout (80% de treino e 20% de teste) (visando a métrica rsme), a segunda tarefa é fazer a mesma abordagem via floresta aleatória (ntrees = 100) e verificar os resultados e a ultima tarefa e realizar um pré - processo com as variaveis e aplicar uma rede neural multilayer perceptron e realizar a mesma abordagem e verificar.

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}

library(caret); library(dplyr); library(datasets); library(psych);

```

### Entrada de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(mtcars)
dim(mtcars)
str(mtcars)
```

### Transformação do tipo de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
mtcars$cyl = as.factor(mtcars$cyl)
mtcars$vs = as.factor(mtcars$vs)
mtcars$am = as.factor(mtcars$am)
mtcars$gear = as.factor(mtcars$gear)
mtcars$carb = as.factor(mtcars$carb)
```

### Transformação de dados
Transformação das variáveis categoricas em variáveis dammys
```{r, cache=FALSE, message=FALSE, warning=FALSE}


```

### Analise exploratoria de dados
Gráficos de histograma:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
par(mfrow=c(3,3))
for(i in 1:12) {
  hist(mtcars[,i], main=names(mtcars)[i])
}
```

Painel de estatísticas:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pairs.panels(mtcars[1:6], gap = 0, bg = c("red", "green", "blue")[mtcars$cyl],pch = 21)
```

Tabelas cruzadas:
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Matriz de correlações:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
corrplot(mtcars,number.digits = 2, number.cex = 0.75)
```

### Seleção de variáveis
Verificar se temos variaveis preditoras com correlação acima de 0.70
```{r, cache=FALSE, message=FALSE, warning=FALSE}
mtcars.cor <- mtcars[, sapply(mtcars, is.numeric)]
mtcars.cor$mpg <- NULL
mtcars.cor <- cor(mtcars.cor)
findCorrelation(mtcars.cor, cutoff = .70, verbose = T, names = T)
```


### Controle do treinamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ctrl = trainControl(method = “cv”,number = 10)
```

### Partição da base
Particionando a base com 80% de treino e 20% de teste.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(25441)
part = createDataPartition(y = mtcars$mpg, p = 0.8, list = FALSE)
treino = mtcars[part,]
teste = mtcars[-part,]
```

### Modelos de regressão linear Hold-out (Treino e resultados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
model_linear = train(mpg~., data = treino, method = "lm")
summary(model_linear)
plot(resid(model_linear))
plot(varImp(model_linear))
```

### Modelos de regressão linear Hold-out (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_rl = pred(model_linear,teste)
```

### Modelos de regressão com base no Random Forest Hold-out (Treino)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
model_rf = train(mpg~., data = treino, method = "rf")
summary(model_rf)
plot(resid(model_rf))
plot(varImp(model_rf))
```

### Modelos de regressão com base no Random Forest Hold-out (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_rl = pred(model_rf,teste)
```
### Modelos de regressão linear Cross Validation 10 (Treino e resultados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_linear = train(mpg~., data = treino, method = "lm", trControl = ctrl, metric="Rsquared")
summary(cv_model_linear)
plot(resid(cv_model_linear))
plot(varImp(cv_model_linear))
```

### Modelos de regressão linear Cross Validation 10 (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_rl = pred(cv_model_linear,teste)
```

### Modelos de regressão com base no Random Forest Cross Validation 10 (Treino)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_rf = train(mpg~., data = treino, method = "rf", trControl = ctrl, metric="Rsquared")
summary(cv_model_rf)
plot(resid(cv_model_rf))
plot(varImp(cv_model_rf))
```

### Modelos de regressão com base no Random Forest Cross Validation 10 (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_pred_rf = pred(cv_model_rf,teste)
```
### Conclusões
