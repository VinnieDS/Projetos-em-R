# Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base BreastCancer.

O objetivo é identificar cada uma das várias classes benignas ou malignas. As amostras chegam periodicamente como o Dr. Wolberg relata seus casos clínicos. O banco de dados, portanto, reflete essa ordem cronológica agrupamento dos dados. Esta informação de agrupamento aparece imediatamente abaixo, tendo sido removida dos dados em si. Cada variável, exceto a primeira, foi convertida em 11 números numéricos primitivos. atributos com valores que variam de 0 a 10. Há 16 valores de atributos ausentes. Uma base de dados com 699 observações em 11 variáveis, sendo uma delas uma variável de caráter, 9 sendo ordenada ou nominal e 1 classe alvo.

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
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(rpart);
library(e1079);
library(Amelia);
library(mlbench);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data(BreastCancer)
dim(BreastCancer)
data = BreastCancer
```

### Análise explorátoria de dados.

* Análise dos dados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(data,Target="Class",op_file = "EDA_BreastCancer.html")
```

### Features Engineering.

* Transformação dos fatores em dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
BreastCancer$Cl.thickness = as.numeric(BreastCancer$Cl.thickness)
BreastCancer$Cell.size = as.numeric(BreastCancer$Cell.size)
BreastCancer$Cell.shape = as.numeric(BreastCancer$Cell.shape)
BreastCancer$Marg.adhesion = as.numeric(BreastCancer$Marg.adhesion)
BreastCancer$Epith.c.size = as.numeric(BreastCancer$Epith.c.size)
BreastCancer$Bare.nuclei = as.numeric(BreastCancer$Bare.nuclei)
BreastCancer$Bl.cromatin = as.numeric(BreastCancer$Bl.cromatin)
BreastCancer$Normal.nucleoli = as.numeric(BreastCancer$Normal.nucleoli)
BreastCancer$Mitoses = as.numeric(BreastCancer$Mitoses)
```
* Verificação dos dados faltantes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
bare_nuclei_miss = rpart(Bare.nuclei ~ Cl.thickness + Cell.size + Cell.shape + Marg.adhesion + Epith.c.size + Epith.c.size + Bl.cromatin + Normal.nucleoli, data=full[!is.na(data$Bare.nuclei),], method="anova")
               
data$Bare.nuclei[is.na(data$Bare.nuclei)] = predict(bare_nuclei_miss, data[is.na(data$Bare.nuclei),])
```
### Análise explorátoria de dados.

* Análise de dados modelados e com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(data,Target="Class",op_file = "EDA_BreastCancer_trans.html")
```
* Matriz de correlação
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```
