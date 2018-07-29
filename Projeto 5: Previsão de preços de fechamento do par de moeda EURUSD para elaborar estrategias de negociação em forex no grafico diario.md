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

### Criação de lag
```{r, cache=FALSE, message=FALSE, warning=FALSE}
rsi = c(NA,head(rsi,-1)) 
adx$ADX = c(NA,head(adx$ADX,-1)) 
trend = c(NA,head(trend,-1))
```

### Construção do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = cbind(dataset,rsi,adx,sar,trend,target)
```

### Retirar NA
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = na.omit(dataset)
```

### EDA
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumStat(dataset,by="A",gp="target",Qnt=seq(0,1,0.1),MesofShape=1,Outlier=TRUE,round=4)
```

```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpNumViz(dataset,gp="target",type=1,nlim=NULL,col=c("blue","yellow","orange"),Page=c(2,2),sample=8)
```

# Seleção de dados 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% select(rsi,ADX,trend,sar,target)
```
