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

## Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) aplicando um modelo de SVM na base Breast Cancer.

O objetivo é identificar cada uma das várias classes benignas ou malignas. As amostras chegam periodicamente como o Dr. Wolberg relata seus casos clínicos. O banco de dados, portanto, reflete essa ordem cronológica agrupamento dos dados. Esta informação de agrupamento aparece imediatamente abaixo, tendo sido removida dos dados em si. Cada variável, exceto a primeira, foi convertida em 11 números numéricos primitivos. atributos com valores que variam de 0 a 10. Há 16 valores de atributos ausentes. Uma base de dados com 699 observações em 11 variáveis, sendo uma delas uma variável de caráter, 9 sendo ordenada ou nominal e 1 classe alvo.

## Projeto 8:  Criação de um modelo de regressão (deep learning) para estimar a Força de Compressão do concreto (FCC). 

O concreto é o material mais importante em engenharia civil. A resistência à compressão do concreto é uma função altamente não-linear da idade e dos ingredientes.

http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength

## Projeto 9: Dados do Banco Mundial (1960 a 2016) Analise explorátoria de dados, criação de target, analise de cluster, o desenvolvimento de um modelo preditivo e visualização.

População de países, taxa de fertilidade e expectativa de vida. 

https://www.kaggle.com/gemartin/world-bank-data-1960-to-2016

## Projeto 10: Projeto 10: Modelo de renovação do seguro de motos comparação entre o modelo de regressão logística e árvores de decisão CART(Problema de classes desbalanceadas).

