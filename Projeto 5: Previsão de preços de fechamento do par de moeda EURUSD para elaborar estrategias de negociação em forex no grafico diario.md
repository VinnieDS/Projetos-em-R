### Previsão de preços de fechamento do par de moeda EURUSD para elaborar estrategias de negociação em forex no gráfico diario

Um estudo para verificação de padrões no gráfico diario no par de moeda EURUSD para criação de estrategias de negociação em forex.

### Pacotes
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(caret);library(smartEAD);library(TTR);library(readr);
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
```

### Lag
```{r, cache=FALSE, message=FALSE, warning=FALSE}
rsi = c(NA,head(rsi,-1)) 
adx$ADX = c(NA,head(adx$ADX,-1)) 
trend = c(NA,head(trend,-1))
```

### Construção do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = cbind(dataset,rsi,adx,sar,trend,target)
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
ExpNumViz(dataset,gp="target",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```

### Seleção de dados 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% select(rsi,ADX,trend,sar,target)
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

### Modelo naive bayes com validacao cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_nb = train(target~., data = treino, method = "nb", trainControl = ctrl)
print(cv_model_nb)

pred_nb = predict(cv_model_nb,teste)

confusionMatrix(pred_nb,teste$target, positive = "UP")
```

### Modelo SVM com validacao cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_svm = train(target~., data = treino, method = "svmLinear2", trainControl = ctrl)
print(cv_model_svm)

pred_svm = predict(cv_model_svm,teste)

confusionMatrix(pred_svm,teste$target, positive = "UP")
```

### Modelo random forest com validacao cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_rf = train(target~., data = treino, method = "rf", trainControl = ctrl)
print(cv_model_rf)

pred_rf = predict(cv_model_rf,teste)

confusionMatrix(pred_rf,teste$target, positive = "UP")
```

### Modelo Xgboost com validacao cruzada
```{r, cache=FALSE, message=FALSE, warning=FALSE}
cv_model_xgbTree = train(target~., data = treino, method = "xgbTree", trainControl = ctrl)
print(cv_model_xgbTree)

pred_xgbTree = predict(cv_model_xgbTree,teste)

confusionMatrix(pred_xgbTree,teste$target, positive = "UP")
```
