# Projeto 9: Store Item Demand Forecasting Challenge - Predict 3 months of item sales at different stores

# Pacotes.

library(dplyr);
library(forecast);
library(caret);
library(SmartEDA);

# Entrada de dados.

treino = read.csv("train.csv")
teste = read.csv("teste.csv")
teste$sales = NA
treino$id = NA
data = rbind(data,teste)
dim(data)
str(data)

# Transformação de dados.

data$date = as.Date(data$date)
data$store = as.factor(data$store)
data$item = as.factor(data$item)

# Features Engineering.
# Criação de informações em reverencia a data.

weekday = weekday(data$date)
month = month(data$date)
year = year(data$date)

# Analise explotoria de dados.

ExpReport(data,Target="sales",op_file = "EDA_base_forecast.html")

# Analise de series de temporais.

data_ts = ts(data$sales, start=c(2013,1,1), end=c(2017,12,31), frequency=365)
plot(data_ts, main = "Sales - Time series")

# Decomposição da serie temporal.

stlRes = stl(data_ts, s.window = "periodic")
plot(stlRes,main = "Decomposição da serie Sales")

# Divisão do dataset

set.seed(8674)
part = createDataPartition(y = data$sales, p = 0.8, list = FALSE)
treino = data[part,]
teste = data[-part,]
