## Projeto 1: Modelo de regressão para estimar o consumo de combustivel na base mtcars

Este conjunto de dados é uma versão ligeiramente modificada do conjunto de dados fornecido na biblioteca do StatLib. Em consonância com o uso de Ross Quinlan (1993) na previsão do atributo "mpg", 8 das instâncias originais foram removidas porque tinham valores desconhecidos para o atributo "mpg". O conjunto de dados original está disponível no arquivo "auto-mpg.data-original".

"Os dados dizem respeito ao consumo de combustível do ciclo urbano em milhas por galão, a ser previsto em termos de 3 atributos discretos e 5 contínuos de valor múltiplo." (Quinlan, 1993)

As tarefas são verificar na base de dados apresentado é possivel prever o consumo de combustível de cada carro (mpg) via modelo de regressão linear comparando o método de reamostragem de validação cruzada (10) com a validação holdout (80% de treino e 20% de teste) (visando a métrica rsme), a segunda tarefa é fazer a mesma abordagem via floresta aleatória (ntrees = 100) e verificar os resultados e a ultima tarefa e realizar um pré - processo com as variaveis e aplicar uma rede neural multilayer perceptron e realizar a mesma abordagem e verificar.

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}

library(caret); library(dplyr); library(datasets); library(psych); library(h2o)

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

Gráficos de boxplot:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
par(mfrow=c(3,3))
for(i in 1:12) {
  boxplot(mtcars[,i], main=names(mtcars)[i])
}
```
Painel de estatísticas:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pairs.panels(mtcars[1:6], gap = 0, bg = c("red", "green", "blue")[mtcars$cyl],pch = 21)
```
Matriz de correlações:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
corrplot(mtcars,number.digits = 2, number.cex = 0.75)
```

### Partição da base
Particionando a base com 80% de treino e 20% de teste.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(25441)
part = createDataPartition(y = mtcars$mpg, p = 0.8, list = FALSE)
treino = mtcars[part,]
teste = mtcars[-part,]
```
### Modelos de regressão linear (Treino e resultados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
model_linear = train(mpg~., data = treino, method = "lm")
summary(model_linear)
residuos = resid(model_linear)
```
### Modelos de regressão linear (Teste)

### Modelos de regressão com base no Random Forest (Treino)

### Modelos de regressão com base no Random Forest (Teste)

### Conclusões
