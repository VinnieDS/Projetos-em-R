# Projeto 7: Desenvolver um modelo de classificação via redução de dimensionalidade (PCA) na base BreastCancer.

O objetivo é identificar cada uma das várias classes benignas ou malignas. As amostras chegam periodicamente como o Dr. Wolberg relata seus casos clínicos. O banco de dados, portanto, reflete essa ordem cronológica agrupamento dos dados. Esta informação de agrupamento aparece imediatamente abaixo, tendo sido removida dos dados em si. Cada variável, exceto a primeira, foi convertida em 11 números numéricos primitivos. atributos com valores que variam de 0 a 10. Há 16 valores de atributos ausentes. Uma base de dados com 699 observações em 11 variáveis, sendo uma delas uma variável de caráter, 9 sendo ordenada ou nominal e 1 classe alvo.

library(mlbench)
