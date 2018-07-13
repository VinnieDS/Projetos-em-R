# Projetos de Machine Learning em linguagem R.

Neste repositorio tem os meus projetos e codigos de aprendizado de maquina em R

## Projeto 1: Modelo de regressão para estimar o consumo de combustivel na base mtcars

Verificar na base mtcars se de acordo com os dados apresentados é possivel estimar com rsme razoável o consumo de combustível (mpg) com o metodo de reamostragem de validação cruzada (10) em modelos de regressão linear e modelos de regressão com base em árvores de decisão

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}

library(caret); library(dplyr); library(datasets); library(psych)

```
### Entrada de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(mtcars)
dim(mtcars)
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

## Transformação de dados

Visando uma melhor performance no modelo criaremos variaveis dummys dos preditores 

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

### Modelos de regressão com base em arvores de decisão (Treino)

### Modelos de regressão com base em arvores de decisão (Teste)

### Métrica RSME

### Conclusões
