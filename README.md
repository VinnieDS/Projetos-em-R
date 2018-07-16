# Projetos de Machine Learning em R e H2O.

Neste repositorio encontra-se os meus projetos realizados e códigos de aprendizado de máquina e aprendizagem profunda em R e H2O e a fonte de dados vem da UCI repositorios de base de Machine learning, Yahoo Finance e Kaggle.

## Projeto 1: Criação de um modelo de regressão (linear, multilayer perceptron e random forest) para estimar o consumo de combustivel na base Motor Trend Car Road Tests (mtcars).

Este conjunto de dados é uma versão ligeiramente modificada do conjunto de dados fornecido na biblioteca do StatLib. Em consonância com o uso de Ross Quinlan (1993) na previsão do atributo "mpg", 8 das instâncias originais foram removidas porque tinham valores desconhecidos para o atributo "mpg". O conjunto de dados original está disponível no arquivo "auto-mpg.data-original".

"Os dados dizem respeito ao consumo de combustível do ciclo urbano em milhas por galão, a ser previsto em termos de 3 atributos discretos e 5 contínuos de valor múltiplo." (Quinlan, 1993)

As tarefas são verificar na base de dados apresentado é possivel prever o consumo de combustível de cada carro (mpg) via modelo de regressão linear comparando o método de reamostragem de validação cruzada (10) com a validação holdout (80% de treino e 20% de teste) (visando a métrica rsme), a segunda tarefa é fazer a mesma abordagem via floresta aleatória (ntrees = 100) e verificar os resultados e a ultima tarefa e realizar um pré - processo com as variaveis e aplicar uma rede neural multilayer perceptron e realizar a mesma abordagem e verificar.

## Projeto 2: Criação de um modelo para detecção de anomalias em pneus de caminhões rodoviarios via Autoenconder R.H2O.



## Projeto 3: Titanic - Machine Learning from Disaster (Kaggle) abordargem com modelos de redes neurais multilayer perceptron com R e H2O.

O naufrágio do RMS Titanic é um dos mais infames naufrágios da história. Em 15 de abril de 1912, durante sua viagem inaugural, o Titanic afundou depois de colidir com um iceberg, matando 1502 de 2224 passageiros e tripulantes. Esta tragédia sensacional chocou a comunidade internacional e levou a melhores normas de segurança para os navios.
Uma das razões pelas quais o naufrágio causou tal perda de vida foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Embora houvesse algum elemento de sorte envolvido na sobrevivência do naufrágio, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta.
Neste desafio, pedimos que você conclua a análise de quais tipos de pessoas provavelmente sobreviveriam.
Para esse desafio irei abordar um modelo de classificação via redes neurais multilayer perceptron.

## Projeto 4: Criação de um modelo de classificação sobre a frequencia de doadores de sangue na cidade de Hsin-Chu, em Taiwan.

Para demonstrar o modelo de marketing RFMTC (uma versão modificada do RFM), este estudo adotou o banco de dados de doadores do Centro de Serviços de Transfusão de Sangue na cidade de Hsin-Chu, em Taiwan. O centro passa seu ônibus de serviço de transfusão de sangue para uma universidade na cidade de Hsin-Chu para coletar sangue doado a cada três meses. Para construir um modelo FRMTC, selecionamos aleatoriamente 748 doadores do banco de dados do doador. Esses 748 dados de doadores, cada um incluindo R (Recência - meses desde a última doação), F (Frequência - número total de doações), M (monetária - total de sangue doado em cc), T (tempo - meses desde a primeira doação) e uma variável binária representando se doou sangue em março de 2007 (1 representa doar sangue; 0 significa não doar sangue).
A abordagem de um modelo Xgboost para classificar os doadores.

## Projeto 5: Previsão de preços de fechamento do par de moeda EURUSD para elaborar estrategias de negociação em forex no grafico diario.

## Projeto 6: Criação de regras de associação para criação de perfis de consumo de produtos do varejo.

## Projeto 7: Forecast do valor de mercado do par de moeda BRLUSD com LSTM e analise de sentimentos do mercado após a eleição presidencial do Brasil.
