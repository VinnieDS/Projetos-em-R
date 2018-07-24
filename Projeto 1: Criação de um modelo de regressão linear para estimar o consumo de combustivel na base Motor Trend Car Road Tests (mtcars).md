## Projeto 1: Modelo de regressão para estimar o consumo de combustivel na base mtcars

Este conjunto de dados é uma versão ligeiramente modificada do conjunto de dados fornecido na biblioteca do StatLib. Em consonância com o uso de Ross Quinlan (1993) na previsão do atributo "mpg", 8 das instâncias originais foram removidas porque tinham valores desconhecidos para o atributo "mpg". O conjunto de dados original está disponível no arquivo "auto-mpg.data-original".

"Os dados dizem respeito ao consumo de combustível do ciclo urbano em milhas por galão, a ser previsto em termos de 3 atributos discretos e 5 contínuos de valor múltiplo." (Quinlan, 1993)

Desenvolver um modelo de regressão linear selecionando as variáveis via stepwise e depois de acordo com esse modelo aplicar reamostragem e gerar outros tipos de modelos de regressão para verificar se temos um aumento de performance do que um modelo de regressão linear.

Informações sobre Atributos:

1] mpg = Miles/(US) gallon

2] cyl = Number of cylinders

3] disp = Displacement (cu.in.)

4] hp = Gross horsepower

5] drat = Rear axle ratio

6] wt = Weight (1000 lbs)

7] qsec = 1/4 mile time

8] vs = Engine (0 = V-shaped, 1 = straight)

9] am = Transmission (0 = automatic, 1 = manual)

10] gear = Number of forward gears

11] carb = Number of carburetors

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}

library(caret);library(dplyr);library(datasets);
library(psych);library(car);library(stats);
library(ggplot2);library(MASS);library(car);
library(knitr);library(printr);

```

### Entrada de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(mtcars)
kable(head(mtcars),align = 'c')
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

### Analise exploratoria de dados

Painel de estatísticas:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pairs.panels(mtcars[1,3,4,5], gap = 0, bg = c("red", "green", "blue")[mtcars$cyl],pch = 18)
```

Relação entre as variáveis (mpg x am):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggplot(mtcars, aes(y=mpg, x=factor(am, labels = c("automatic", "manual")), fill=factor(am)))+
        geom_violin(colour="black", size=1)+
        xlab("transmission") + ylab("MPG")
```

Relação entre as variáveis (mpg x cyl):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggplot(mtcars, aes(y=mpg, x=factor(cyl, labels = c("2","4","6")), fill=factor(cyl)))+
        geom_violin(colour="black", size=1)+
        xlab("CYL") + ylab("MPG")

```

### Seleção de variáveis

Procedimento de seleção do modelo
```{r, cache=FALSE, message=FALSE, warning=FALSE}
summary(lm(mpg ~ cyl+disp+hp+drat+wt+qsec+factor(vs)+factor(am)+gear+carb, data = mtcars))$coef
```
Detectando colinearidade
```{r, cache=FALSE, message=FALSE, warning=FALSE}
fitvif = lm(mpg ~ cyl+disp+hp+drat+wt+qsec+factor(vs)+factor(am)+gear+carb, data = mtcars)
kable(vif(fitvif),align = 'c')
```
Método de seleção gradual
```{r, cache=FALSE, message=FALSE, warning=FALSE}
fit = lm(mpg ~ cyl+disp+hp+drat+wt+qsec+factor(vs)+factor(am)+gear+carb, data = mtcars)
step = stepAIC(fit, direction="both", trace=FALSE)
summary(step)$coeff
summary(step)$r.squared
```
Teste de razão de verossimilhança
```{r, cache=FALSE, message=FALSE, warning=FALSE}
fit1 = lm(mpg ~ factor(am), data = mtcars)
fit2 = lm(mpg ~ factor(am)+wt, data = mtcars)
fit3 = lm(mpg ~ factor(am)+wt+qsec, data = mtcars)
fit4 = lm(mpg ~ factor(am)+wt+qsec+hp, data = mtcars)
fit5 = lm(mpg ~ factor(am)+wt+qsec+hp+drat, data = mtcars)
anova(fit1, fit2, fit3, fit4, fit5)
```
Ajustando o modelo final
```{r, cache=FALSE, message=FALSE, warning=FALSE}
modelo_rl_final = lm(mpg ~ wt+qsec+factor(am), data = mtcars)
summary(modelo_rl_final)$coef
summary(modelo_rl_final)$r.squared
```

### Controle do treinamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ctrl = trainControl(method = "cv",number = 10)
```

### Partição da base
Particionando a base com 70% de treino e 30% de teste.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(25441)
part = createDataPartition(y = mtcars$mpg, p = 0.7, list = FALSE)
treino = mtcars[part,]
teste = mtcars[-part,]
```

### Modelos de regressão linear Cross Validation 10 (Treino e resultados)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_linear = train(mpg~wt+qsec+factor(am), data = treino, method = "lm", trainControl = ctrl, metric="Rsquared")
summary(cv_model_linear)
plot(resid(cv_model_linear))
plot(varImp(cv_model_linear))
```

### Modelos de regressão linear Cross Validation 10 (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_rl = predict(cv_model_linear,teste)
```

### Resultados modelo de regressão linear Cross Validation 10
```{r, cache=FALSE, message=FALSE, warning=FALSE}
res_cv_model_linear = data.frame(obs = teste$mpg, pred=pred_rl)
defaultSummary(res_cv_model_linear)
```

### Modelos de regressão com base no Random Forest Cross Validation 10 (Treino)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_rf = train(mpg~wt+qsec+factor(am), data = treino, method = "rf", trainControl = ctrl, metric="Rsquared")
summary(cv_model_rf)
plot(resid(cv_model_rf))
```

### Modelos de regressão com base no Random Forest Cross Validation 10 (Teste)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_pred_rf = predict(cv_model_rf,teste)
```

### Resultados modelo de regressão com base no Random Forest Cross Validation 10
```{r, cache=FALSE, message=FALSE, warning=FALSE}
res_cv_model_rf = data.frame(obs = teste$mpg, pred=cv_pred_rf)
defaultSummary(res_cv_model_rf)
```

### Conclusões
