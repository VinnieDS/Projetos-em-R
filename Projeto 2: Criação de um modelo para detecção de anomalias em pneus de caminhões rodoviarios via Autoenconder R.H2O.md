## Projeto 2: Criação de um modelo para detecção de anomalias em pneus de caminhões rodóviarios via Autoenconder R.H2O.

Na área de manutenção de caminhões recebe as informações de maneira remota sobre as rotas dos caminhões e os dados de telemetria dos pneus. Depois de muitos defeitos nos pneus de caminhões e de até acidentes o pedido da área de manutenção para área de inteligencia criar um modelo de detecção de anomalias para evitar possíveis defeitos e acidentes mesclando os dados do caminhão e dos dados de telemetria. De acordo com os dados apresentados modelar um dataset juntando os dados do caminhão e da telemetria e depois criar um modelo de deep learning (Autoenconder) no framework H2O.ai e verificar os dados anomalos gerados e tambem verificar a importância das rotas nas anomalias geradas no modelo e gerar um sistema para indicar essas anomalias.

### Pacotes.
```{r, cache=FALSE, message=FALSE, warning=FALSE}
library(h2o);library(dplyr);library(ggplot2);library(MASS);
```

### Entrada de dados.

Dados de telemetria

Dados de rotas

### Modelagem do dataset com práticas de ETL em dplyr.

### Analise exploratoria de dados.
