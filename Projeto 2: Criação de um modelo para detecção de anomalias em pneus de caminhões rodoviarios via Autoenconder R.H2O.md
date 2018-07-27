## Projeto 2: Criação de um modelo para detecção de anomalias em pneus de caminhões rodóviarios via Autoenconder R.H2O.

Na área de manutenção de caminhões recebe as informações de maneira remota sobre as rotas dos caminhões e os dados de telemetria dos pneus. Depois de muitos defeitos nos pneus de caminhões e de até acidentes o pedido da área de manutenção para área de inteligencia criar um modelo de detecção de anomalias para evitar possíveis defeitos e acidentes mesclando os dados do caminhão e dos dados de telemetria. De acordo com os dados apresentados modelar um dataset juntando os dados do caminhão e da telemetria e depois criar um modelo de deep learning (Autoenconder) no framework H2O.ai e verificar os dados anomalos gerados e tambem verificar a importância das rotas nas anomalias geradas no modelo e gerar um sistema para indicar essas anomalias.

### Pacotes.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(h2o);library(dplyr);library(ggplot2);library(caret);library(stringr);
```

### Entrada de dados.
Dados de telemetria e dos caminhões
```{r, cache=FALSE, message=FALSE, warning=FALSE}
mydata1 = read.csv2("tele.csv")
str(mydata1)
```

Dados de rotas dos caminhões
```{r, cache=FALSE, message=FALSE, warning=FALSE}
mydata2 = read.csv2("rota.csv")
str(mydata2)
```

### Modelagem do dataset em dplyr.
Seleção de variáveis:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
base_telemetria = mydata1 %>% select(h_tyredatakey,vehicleid,tyreserialnumber,mrxWheelNumber,wheelpositionname,td_time,td_press,td_temp)
str(base_telemetria)

base_rotas = mydata2 %>% select(TRUCK,LOC,BLAST,EXCAV,LOAD,DIST,SHIFT,DDMMYY)
str(base_rotas)
```

Criação do chaveiro (Data & Caminhão):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
chave = str_c(base_telemetria$vehicleid, base_telemetria$td_time)
base_telemetria = cbind(chave,base_telemetria)

chave = str_c(base_rotas$TRUCK, base_rotas$tDDMMYY)
base_rotas = cbind(chave,base_rotas)
```

Junção das base de telemetria e das rotas dos caminhões:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = left_join(base_telemetria,base_rotas,by="chave")
```

Ajustamento dos dados na regra do negócio (retirar instâncias com menos de 600 Bar):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% filter(td_press<600)
```

Criação de variáveis (Verificar os quartis dos dados contínuos):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset.bin = subset(dataset, select = ...)

v.bin  = c("td_temp","td_press")

for (i in 1:length(v.bin) ){ 
  v_x    = dataset[,v.bin[i]]
  qnt    = unique(quantile(v_x,seq(0, 1, .25)))
 
  for ( ii in 1:(length(qnt)-1) ) {
    dataset.bin$tmp   = ifelse (v_x > qnt[ii] & v_x <= qnt[ii+1], 1, 0)
    dataset.bin$tmp   = ifelse (qnt[ii] == qnt[1] & v_x == qnt[ii], 1, dataset.bin$tmp)
    names(dataset.bin)[names(dataset.bin)=="tmp"] = paste(v.bin[i],"_disc_", ii, sep = "")
  }
}

```

Criação de variáveis (Transformação da informação dos trechos):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dummy = dummyVars(~.,data = dataset,levelsOnly = TRUE)
predict(dummy, dataset)
```

Dataset final:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset = dataset %>% select()
```


### Analise exploratoria de dados no dataset

Visão completa do dataset:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ExpReport(dataset,op_file = "teste.html")
```

Gráficos de correlação de variáveis:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
ggcorr(dataset,label = T,nbreaks = 5,label_round = 4)
```

Analise das varivaeis continuas:
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Tabela cruzadas Pressão e Posição da roda:
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Modelagem da rede neural Autoencoder não supervisionado via H2O.

Iniciar o framework h2o no R:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
h2o.init()
```

Inserir os dados no h2o:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset.hex = as.h2o(dataset, destination_frame="dataset.hex")
```

Treino do modelo da rede neural Autoencoder (epochs = 100):
```{r, cache=FALSE, message=FALSE, warning=FALSE}
modelo_auto = h2o.deeplearning(x = feature_names, training_frame = dataset.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 547813,
                               hidden = c(6,5,6), epochs = 100)                         
```

Detecção de anomalias do modelo:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset_anomalias = h2o.anomaly(modelo_auto, dataset.hex, per_feature=FALSE)
head(dataset_anomalias)
tab_anomalias = as.data.frame(dataset_anomalias)
```

### Analise dos resultados da rede neural Autoencoder.

Gráfico de reconstrução do dataset:
```{r, cache=FALSE, message=FALSE, warning=FALSE}
plot(sort(tab_anomalias$Reconstruction.MSE), main='Reconstrução do dataset')
```

Ponto de corte dos dados anomalos:
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Definição das anomalias no dataset.
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Verificação dos trechos com maior frequencia de anomalias detectadas.

### Verificação dos caminhões com maior frequencia de anomalias detectadas.

### Conclusão.

Numa amostra de um mês de telemetria e rotas
