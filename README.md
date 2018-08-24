# Projetos de Machine Learning em R e H2O.

Neste repositório encontram-se os meus projetos realizados e códigos de aprendizado de máquina e aprendizagem profunda em R e H2O em diversos universos de conhecimentos e todas as abordagens de aprendizado supervisionada e não supervisionada.

## Projeto 1: Criação de um modelo de regressão para estimar o consumo de combustivel na base Motor Trend Car Road Tests (mtcars).

Este conjunto de dados é uma versão ligeiramente modificada do conjunto de dados fornecido na biblioteca do StatLib. Em consonância com o uso de Ross Quinlan (1993) na previsão do atributo "mpg", 8 das instâncias originais foram removidas porque tinham valores desconhecidos para o atributo "mpg". O conjunto de dados original está disponível no arquivo "auto-mpg.data-original".
"Os dados dizem respeito ao consumo de combustível do ciclo urbano em milhas por galão, a ser previsto em termos de 3 atributos discretos e 5 contínuos de valor múltiplo." (Quinlan, 1993)

Desenvolver um modelo de regressão linear selecionando as variáveis via stepwise e depois de acordo com esse modelo aplicar reamostragem e gerar outros tipos de modelos de regressão para verificar se temos um aumento de performance do que um modelo de regressão linear.

## Projeto 2: Criação de um modelo para detecção de anomalias em pneus de caminhões rodóviarios via Autoenconder R e H2O.

Na área de manutenção de caminhões recebe as informações de maneira remota sobre as rotas dos caminhões e os dados de telemetria dos pneus. Depois de muitos defeitos nos pneus de caminhões e de até acidentes o pedido da área de manutenção para área de inteligencia criar um modelo de detecção de anomalias para evitar possíveis defeitos e acidentes mesclando os dados do caminhão e dos dados de telemetria. De acordo com os dados apresentados modelar um dataset juntando os dados do caminhão e da telemetria e depois criar um modelo de deep learning (Autoenconder) no framework H2O.ai e verificar os dados anomalos gerados e tambem verificar a importância das rotas nas anomalias geradas no modelo e gerar um sistema para indicar essas anomalias.

## Projeto 3: Titanic - Machine Learning from Disaster (Kaggle) abordargem com modelos de redes neurais multilayer perceptron com R e H2O.

O naufrágio do RMS Titanic é um dos mais infames naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios. Uma das razões pelas quais o naufrágio causou tal perda de vida foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta. Neste desafio, pedimos que você conclua a análise de quais tipos de pessoas provavelmente sobreviveriam. Para esse desafio irei abordar um modelo de classificação via deep learning no h2o, pois estamos falando de um desenvolvimento de modelo não linear.

https://www.kaggle.com/c/titanic/data

## Projeto 4: Criação de um modelo de classificação sobre a frequencia de doadores de sangue na cidade de Hsin-Chu, em Taiwan.

Este estudo adotou o banco de dados de doadores do Centro de Serviços de Transfusão de Sangue na cidade de Hsin-Chu, em Taiwan. O centro passa seu ônibus de serviço de transfusão de sangue para uma universidade na cidade de Hsin-Chu para coletar sangue doado a cada três meses. Para construir um modelo de machine learning, selecionamos aleatoriamente 748 doadores do banco de dados do doador. Esses 748 dados de doadores, cada um incluindo R (Recência - meses desde a última doação), F (Frequência - número total de doações), M (monetária - total de sangue doado em cc), T (tempo - meses desde a primeira doação) e uma variável binária representando se doou sangue em março de 2007 (1 representa doar sangue; 0 significa não doar sangue). De acordo com uma análise explorátoria de dados e um entendimento dos dados podemos abordar um modelo de classificação com base numa árvore de decisão e tambem num modelo Xgboost.

## Projeto 5: Previsão de preços de fechamento do par de moeda EURUSD para elaborar estrategias de negociação em forex no grafico diario.

Um estudo para verificação de padrões no gráfico diario no par de moeda EURUSD para criação de estrategias de negociação em forex.

## Projeto 6: Porto Seguro’s Safe Driver Prediction - Kaggle

Nada estraga a emoção de comprar um carro novo mais rapidamente do que ver sua nova fatura de seguro. A dor é ainda mais dolorosa quando você sabe que é um bom motorista. Não parece justo que você tenha que pagar tanto se for cauteloso durante anos. A Porto Seguro , uma das maiores seguradoras de automóveis e residenciais do Brasil, concorda completamente. Imprecisões nas previsões de sinistro da companhia de seguros de automóveis aumentam o custo do seguro para os bons motoristas e reduzem o preço dos maus. Nesta competição, você é desafiado a construir um modelo que prevê a probabilidade de um motorista iniciar uma reivindicação de seguro de automóvel no próximo ano. Embora a Porto Seguro tenha usado o aprendizado de máquina nos últimos 20 anos, eles procuram a comunidade de aprendizado de máquinas de Kaggle para explorar métodos novos e mais poderosos. Uma previsão mais precisa permitir-lhes-á adaptar ainda mais os seus preços e, esperamos, tornar a cobertura do seguro automóvel mais acessível a mais condutores.

## Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base Breast Cancer.

As características são calculadas a partir de uma imagem digitalizada de um aspirador de agulha fina (PAAF) de uma massa mamária. Eles descrevem características dos núcleos celulares presentes na imagem. O espaço tridimensional é o descrito em: [KP Bennett e OL Mangasarian: "Discriminação Linear de Programação Robusta de Dois Conjuntos Linearmente Inseparáveis", Optimization Methods and Software 1, 1992, 23-34]. Esta base de dados também está disponível através do servidor ftp da UW CS: ftp ftp.cs.wisc.edu cd math-prog / cpo-dataset / machine-learn / WDBC / Também pode ser encontrado no UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Informações sobre Atributos:

1) Número ID 
2) Diagnóstico (M = maligno, B = benigno) 

Dez características reais são calculadas para cada núcleo celular:

a) raio (média das distâncias do centro para os pontos no perímetro) 
b) textura (desvio padrão dos valores da escala de cinza) 
c) perímetro 
d) área 
e) suavidade (variação local no comprimento do raio) 
f) compactação (perímetro ^ 2 / área - 1.0) 
g) concavidade (gravidade das porções côncavas do contorno) 
h) pontos côncavos (número de porções côncavas do contorno)
i) simetria 
j) dimensão fractal ("aproximação costeira" - 1)

A média, erro padrão e "pior" ou maior (média dos três maiores valores) desses recursos foram calculados para cada imagem, resultando em 30 recursos. Por exemplo, o campo 3 é o raio médio, o campo 13 é o raio SE, o campo 23 é o pior raio.

Todos os valores de recursos são recodificados com quatro dígitos significativos.
Valores de atributo ausentes: nenhum
Distribuição de classes: 357 benignas, 212 malignas

https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/home

## Projeto 8: Criação de um modelo de regressão (deep learning) para estimar a Força de Compressão do concreto (FCC). 

O concreto é o material mais importante em engenharia civil. A resistência à compressão do concreto é uma função altamente não-linear da idade e dos ingredientes.

http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength

## Projeto 9: Store Item Demand Forecasting Challenge - Predict 3 months of item sales at different stores - Kaggle.

Esta competição é oferecida como uma maneira de explorar diferentes técnicas de séries temporais em um conjunto de dados relativamente simples e limpo. Você recebe 5 anos de dados de vendas de itens de loja e pede para prever 3 meses de vendas de 50 itens diferentes em 10 lojas diferentes. Qual é a melhor maneira de lidar com a sazonalidade? As lojas devem ser modeladas separadamente ou você pode agrupá-las juntas? O aprendizado profundo funciona melhor que o ARIMA? Pode bater o xgboost? Esta é uma grande competição para explorar diferentes modelos e melhorar suas habilidades em previsão.

https://www.kaggle.com/c/demand-forecasting-kernels-only

## Projeto 10: Modelo de renovação do seguro de motos comparação entre o modelo de regressão logística e árvores de decisão CART (Problema de classes desbalanceadas).

