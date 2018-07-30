### Previsão de preços de fechamento do par de moeda EURUSD para elaborar estrategias de negociação em forex no gráfico diario

Um estudo para verificação de padrões no gráfico diario no par de moeda EURUSD para criação de estrategias de negociação em forex.

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(caret);library(SmartEDA);library(TTR);library(readr);library(GGally);library(forecast);library(dplyr);
```

### Entrada de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = read_delim("EUR_USD.csv", ";", escape_double = FALSE, col_types = cols(Data = col_date(format = "%d.%m.%Y"), Var = col_skip()), trim_ws = TRUE)
```

### Criação de indicadores técnicos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
price = dataset$Close-dataset$Open
target = ifelse(price > 0,"UP","Down")
rsi = RSI(dataset$Close, n=14, maType="WMA")
adx = data.frame(ADX(dataset[,c("Maxima","Minima","Close")]))
sar = SAR(dataset[,c("Maxima","Minima")], accel = c(0.02, 0.2))
trend = dataset$Close - sar
trend_cci = CCI(dataset[,c("Maxima","Minima","Close")])
trend_vhf = VHF(dataset[,c("Maxima","Minima","Close")])
```

### Lag
```{r, cache=FALSE, message=FALSE, warning=FALSE}
rsi = c(NA,head(rsi,-1)) 
adx$ADX = c(NA,head(adx$ADX,-1)) 
trend = c(NA,head(trend,-1))
trend_aroon = c(NA,head(trend_aroon,-1))
trend_cci = c(NA,head(trend_cci,-1))
trend_vhf = c(NA,head(trend_vhf,-1))
```

### Construção do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = cbind(dataset,rsi,adx,sar,trend,trend_aroon,trend_cci,trend_vhf,target)
```

### Retirar as instâncias com dados faltantes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = na.omit(dataset)
```

### Analise de series temporal

Transformação dos dados para uma serie temporal
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset_ts = ts(dataset$Close, start=c(2013,1,1), end=c(2018,5,14), frequency=365)
plot(dataset_ts, main = "EURUSD - 2013-01-01 at 2018-05-14")
```

Decomposição da serie temporal
```{r, cache=FALSE, message=FALSE, warning=FALSE}
stlRes = stl(dataset_ts, s.window = "periodic")
plot(stlRes,main = "Decomposição da serie EURUSD")
```

### Análise exploratoria de dados do dataset

Avaliação dos dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(dataset,by="A",gp="target",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
```
Avalição dos dados númericos no foco do target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumViz(dataset,gp="target",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```
Matriz de correlações
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(dataset,label = T,nbreaks = 5,label_round = 4)
```
### Análise exploratoria de dados dos ciclos

Avaliação dos dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(dataset,by="A",gp="target",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
```
Avalição dos dados númericos no foco do target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumViz(dataset,gp="target",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```
Matriz de correlações
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(dataset,label = T,nbreaks = 5,label_round = 4)
```
### Análise exploratoria de dados dos anos

Avaliação dos dados númericos
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(dataset,by="A",gp="target",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
```
Avalição dos dados númericos no foco do target
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumViz(dataset,gp="target",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```
Matriz de correlações
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(dataset,label = T,nbreaks = 5,label_round = 4)
```
### Hipotese 1

Treinar um modelo boosting (Xgboost) partição aleatoria 

### Seleção de dados 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% select(ADX,sar,trend_cci,trend_vhf,target)
```

### Controle do treinamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ctrl = trainControl(method = "cv",number = 5,verboseIter = TRUE,savePredictions = "final")
```

### Grid search
```{r, cache=FALSE, message=FALSE, warning=FALSE}
xgb_grid = expand.grid(nrounds = c(100, 150, 200),max_depth = 1,min_child_weight = 1,subsample = 1,gamma = 0,colsample_bytree = 0.8,eta = c(.2, .3, .4))
```

### Particionamento de dados

```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(77489)
part = createDataPartition(y = dataset$target, p = 0.9, list = FALSE)
treino = dataset[part,]
teste = dataset[-part,]
```

### Treino e Teste do modelo Xgboost

Treino
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_xgbTree = train(target~., data = treino, method = "xgbTree", trainControl = ctrl, tuneGrid = xgb_grid)
print(cv_model_xgbTree)
confusionMatrix(cv_model_xgbTree,positive = "UP")
plot(cv_model_xgbTree)
plot(varImp(xgboost.model))
```

Teste
```{r, cache=FALSE, message=FALSE, warning=FALSE}
pred_xgbTree = predict(cv_model_xgbTree,teste)
confusionMatrix(pred_xgbTree,teste$target, positive = "UP")

```
### Hipotese 2

Treinar modelos sazonais um para tendencia de alta e um outro para tendencia de baixa no gráfico diario do EURUSD. Na analise de series temporais na amostra podemos visualizar dois ciclos de alta de baixa e assim tentar diminuir o erro de classificação do modelo na hipotese 1.
