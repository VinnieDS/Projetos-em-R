### Previsão de preços de fechamento do par de moeda EURUSD para elaborar estrategias de negociação em forex no gráfico diario

Um estudo para verificação de padrões no gráfico diario no par de moeda EURUSD para criação de estrategias de negociação em forex.

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(caret);library(smartEAD);library(TTR);library(readr);library(GGally);
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
trend_aroon = aroon(dataset[,c("Maxima", "Minima")],n=14)
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

### Análise exploratoria de dados

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

### Seleção de dados 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% select(ADX,sar,trend_cci,trend_vhf,target)
```
### Controle do treinamento
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ctrl = trainControl(method = "cv",number = 5)
```

### Particionamento de dados
```{r, cache=FALSE, message=FALSE, warning=FALSE}
set.seed(77489)
part = createDataPartition(y = dataset$target, p = 0.8, list = FALSE)
treino = dataset[part,]
teste = dataset[-part,]
```

### Modelo Naive Bayes com validação cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_nb = train(target~., data = treino, method = "nb", trainControl = ctrl)
print(cv_model_nb)

pred_nb = predict(cv_model_nb,teste)

confusionMatrix(pred_nb,teste$target, positive = "UP")
```

### Modelo SVM com validação cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_svm = train(target~., data = treino, method = "svmLinear2", trainControl = ctrl)
print(cv_model_svm)

pred_svm = predict(cv_model_svm,teste)

confusionMatrix(pred_svm,teste$target, positive = "UP")
```

### Modelo Random Forest com validação cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_rf = train(target~., data = treino, method = "rf", trainControl = ctrl)
print(cv_model_rf)

pred_rf = predict(cv_model_rf,teste)

confusionMatrix(pred_rf,teste$target, positive = "UP")
```

### Modelo Xgboost com validação cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_xgbTree = train(target~., data = treino, method = "xgbTree", trainControl = ctrl)
print(cv_model_xgbTree)

pred_xgbTree = predict(cv_model_xgbTree,teste)

confusionMatrix(pred_xgbTree,teste$target, positive = "UP")
```
