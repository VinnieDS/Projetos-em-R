## Projeto 2: Criação de um modelo para detecção de anomalias em pneus de caminhões rodóviarios via Autoenconder R.H2O.

Na área de manutenção de caminhões recebe as informações de maneira remota sobre as rotas dos caminhões e os dados de telemetria dos pneus. Depois de muitos defeitos nos pneus de caminhões e de até acidentes o pedido da área de manutenção para área de inteligencia criar um modelo de detecção de anomalias para evitar possíveis defeitos e acidentes mesclando os dados do caminhão e dos dados de telemetria. De acordo com os dados apresentados modelar um dataset juntando os dados do caminhão e da telemetria e depois criar um modelo de deep learning (Autoenconder) no framework H2O.ai e verificar os dados anomalos gerados e tambem verificar a importância das rotas nas anomalias geradas no modelo e gerar um sistema para indicar essas anomalias.

### Pacotes.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(h2o);library(dplyr);library(ggplot2);library(caret);
```

### Entrada de dados.
Dados de telemetria e dos caminhões
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Dados de rotas dos caminhões
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Modelagem do dataset com práticas de ETL em dplyr.
Seleção de variáveis
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Junção das base de telemetria e das rotas dos caminhões
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Ajustamento dos dados na regra do negócio 
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Criação de variáveis (Classificação estatística dos dados númericos)
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

Criação de variáveis (Transformação da informação dos trechos)
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Analise exploratoria de dados no dataset

Historgramas
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Modelagem da rede neural Autoencoder não supervisionado via H2O.

Iniciar o framework h2o no R
```{r, cache=FALSE, message=FALSE, warning=FALSE}
h2o.init()
```

Inserir os dados no h2o
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset.hex = as.h2o(dataset, destination_frame="dataset.hex")
```

Modelo de rede neural Autoencoder (epochs = 100)
```{r, cache=FALSE, message=FALSE, warning=FALSE}
modelo_auto = h2o.deeplearning(x = feature_names, training_frame = dataset.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 547813,
                               hidden = c(6,5,6), epochs = 100)                         
```

Detecção de anomalias do modelo 
```{r, cache=FALSE, message=FALSE, warning=FALSE}
dataset_anomalias = h2o.anomaly(modelo_auto, dataset.hex, per_feature=FALSE)
head(dataset_anomalias)
tab_anomalias = as.data.frame(dataset_anomalias)
```

### Analise dos resultados da rede neural Autoencoder.

Gráfico de reconstrução do dataset
```{r, cache=FALSE, message=FALSE, warning=FALSE}
plot(sort(tab_anomalias$Reconstruction.MSE), main='Reconstruction Error')
```

### Definição das anomalias no dataset.
```{r, cache=FALSE, message=FALSE, warning=FALSE}

```

### Sinalização no dataset das anomalias.

### Verificação dos trechos com maior frequencia de anomalias detectadas.

### Verificação dos caminhões com maior frequencia de anomalias detectadas.

### Conclusão.
