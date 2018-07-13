# Projetos de Machine Learning em linguagem R.

Neste repositorio tem os meus projetos e codigos de aprendizado de maquina em R

## Projeto 1: Modelo de regressão para estimar o consumo de combustivel na base mtcars

Verificar na base mtcars se de acordo com os dados apresentados é possivel estimar com rsme razoável o consumo de combustível (mpg) com o metodo de reamostragem de validação cruzada (10) em modelos de regressão linear e modelos de regressão com base em árvores de decisão

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}

library(caret); library(dplyr); library(ggplot2); library(datasets); library(psych)

```
### Entrada de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}

data(mtcars)

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
### Reamostragem

### Modelos de regressão linear (Treino)

### Modelos de regressão com base em arvores de decisão (Treino)

### Modelos de regressão linear (Teste)

### Modelos de regressão com base em arvores de decisão (Teste)

### Métrica RSME

### Conclusões
