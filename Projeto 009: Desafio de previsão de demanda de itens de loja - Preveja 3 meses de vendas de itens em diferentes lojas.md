# Projeto 9: Desafio de previsão de demanda de itens de loja - Preveja 3 meses de vendas de itens em diferentes lojas

Esta competição é oferecida como uma maneira de explorar diferentes técnicas de séries temporais em um conjunto de dados relativamente simples e limpo. Você recebe 5 anos de dados de vendas de itens de loja e pede para prever 3 meses de vendas de 50 itens diferentes em 10 lojas diferentes. Qual é a melhor maneira de lidar com a sazonalidade? As lojas devem ser modeladas separadamente ou você pode agrupá-las juntas? O aprendizado profundo funciona melhor que o ARIMA? Pode bater o xgboost? Esta é uma grande competição para explorar diferentes modelos e melhorar suas habilidades em previsão.

### Pacotes.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(dplyr);
library(forecast);
library(caret);
library(SmartEDA);
```

### Entrada de dados.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
treino = read.csv("train.csv")
teste = read.csv("teste.csv")
teste$sales = NA
treino$id = NA
data = rbind(data,teste)
dim(data)
str(data)
```

### Transformação de dados.

* Transformação do tipo de dados.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data$date = as.Date(data$date)
data$store = as.factor(data$store)
data$item = as.factor(data$item)
```

* Criação de informações em reverencia a data.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
# Dia de semana
weekday = weekday(data$date)

# Mês
month = as.numeric(format(date, format = "%m"))

# Ano
year = as.numeric(format(date, format = "%Y"))

# Trimestre
month = as.numeric(format(date, format = "%m"))[1]
if (month < 4) {
    quarter = paste( format(date, format = "%Y")[1], "Q1", sep="-")
} else if (month > 3 & month < 7) {
    quarter = paste( format(date, format = "%Y")[1], "Q2", sep="-")            
} else if (month > 6 & month < 10) {
    quarter = paste( format(date, format = "%Y")[1], "Q3", sep="-")
} else if (month > 9) {
    quarter = paste( format(date, format = "%Y")[1], "Q4", sep="-")
}
```

### Análise explotoria de dados.

* Analise dos dados com foco do target.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(data,Target="sales",op_file = "EDA_base_forecast.html")
```

### Análise de series de temporais.

* Gráfico da serie temporal.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
data_ts = ts(data$sales, start=c(2013,1,1), end=c(2017,12,31), frequency=365)
plot(data_ts, main = "Sales - Time series")
```

* Decomposição da serie temporal.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
stlRes = stl(data_ts, s.window = "periodic")
plot(stlRes,main = "Decomposição da serie Sales")
```

### Features Engineering.

* Agregar as vendas via média.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
```

* Agregar as vendas via desvio padrão.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
```

* Agregar as vendas via mediana.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
```

### Divisão do dataset.

```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(8674)
part = createDataPartition(y = data$sales, p = 0.90, list = FALSE)
treino = data[part,]
teste = data[-part,]
```
