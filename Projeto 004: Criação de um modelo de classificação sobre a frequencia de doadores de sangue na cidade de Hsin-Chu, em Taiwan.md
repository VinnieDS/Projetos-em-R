# Projeto 4: Criação de um modelo de classificação sobre a frequencia de doadores de sangue na cidade de Hsin-Chu, em Taiwan

Este estudo adotou o banco de dados de doadores do Centro de Serviços de Transfusão de Sangue na cidade de Hsin-Chu, em Taiwan. O centro passa seu ônibus de serviço de transfusão de sangue para uma universidade na cidade de Hsin-Chu para coletar sangue doado a cada três meses. Para construir um modelo de machine learning, selecionamos aleatoriamente 748 doadores do banco de dados do doador. Esses 748 dados de doadores, cada um incluindo R (Recência - meses desde a última doação), F (Frequência - número total de doações), M (monetária - total de sangue doado em cc), T (tempo - meses desde a primeira doação) e uma variável binária representando se doou sangue em março de 2007 (1 representa doar sangue; 0 significa não doar sangue). De acordo com uma análise explorátoria de dados e um entendimento dos dados podemos abordar um modelo de classificação com base numa árvore de decisão e tambem num modelo Xgboost.

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr);
library(caret);
library(SmartEDA);
library(GGally);
library(Matrix);
library(Xgboost);
library(rpart);
```

### Entrada de dados.

* Dados de treino e teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
datadb = read.csv2("datadb.csv")
dim(datadb)
```

### Análise explorátoria de dados.

* Análise dos dados brutos com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(datadb,Target="class",op_file = "EDA_csts.html")
```

### Features Engineering.

* Transformação dos dados númericos em fatores
```{r, cache=FALSE, message=FALSE, warning=FALSE}
datadb$class = as.factor(datadb$class)
```

### Análise explorátoria de dados.

* Análise dos dados modelados com foco no target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(datadb,Target="class",op_file = "EDA_csts.html")
```
* Matriz de correlação com o dado pre processado
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Preparação para o treinamento.

* Pré-processamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```
