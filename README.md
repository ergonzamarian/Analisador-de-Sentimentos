# Analisador-de-Sentimentos
Realiza a análise de comentários de filmes, produtos eletrônicos e outros canais através de Inteligência Artificial.

----------------------
DESENVOLVEDOR E VERSÃO
----------------------

Nome: Ergon Zamarian Lima
Versão do python utilizada: Python 3.6.4
 
---------------------------------
ANÁLISE E SENTIMENTO - EXPLICAÇÃO
---------------------------------

# Resumo

Esta implementação demonstra de forma simplificada o funcionamento de um analisador de sentimentos que se utiliza de uma base de dados já previamente fornecida, com saída informando se o comentário é Positivo ou Negativo. Será mostrado também 
a Acurácia, Precisão, Recall e F1 Score na tela de execução onde o cálculo dos mesmos é feito da seguinte forma:

* Fórmula

- Acurácia = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + 	   		      	     alsos_positivos + falsos_negativos)
- Precisão = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
- Recall   = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
- F1 Score = 2*(Recall * precisao) / (Recall + Precisão)


Para medirmos o resultado, utiliza-se um percentual da base para treinamento e outro percentual para a avaliação dos resultados.
-> 75% dos registros para treino e 25% para validação
Como identifica-se o sentimento passado através de um texto?
-> Transforma-se todos os comentários em uma lista de valores númericos que representam a frequência da cada palavra, Com essas frequências calculamos a pontuação para sentimentos positivos e para sentimentos negativos para cada uma das palavras, e por fim, o comentário completo.
-> Utilizou-se para realizar tal procedimento o "CountVectorizer" importado do sklearn.feature_extraction.text e BernoulliNB do pacote sklearn.naive_bayes.
-> Separa-se todos os registros de treino em duas listas, uma contendo o texto e a outra contendo as respostas.
-> Depois utiliza-se o CountVectorizer para criar a representação de frequência e através dela gerar nosso classificador.
-> Substitui-se os textos pela sua representação de frequência, e utiliza-se do método fit presente em BernoulliNB para gerar o modelo classificador.
-> A partir deste ponto já estamos com nosso modelo pronto para avaliar novos comentários e com nosso Array criado com 1 para comentários positivos e 0 para negativos.
-> Para a estratégia de validção teremos que percorrer todos os registros que separamos para teste e compararmos o resultado real com o resultado obtido pelo modelo que criamos, contabilizando assim os acertos.
-> Depois dessa etapa podemos por fim extrair nossas 4 métricas utilizadas para o cálculo da fórmula acima (Acurácia, Precisão, Recall e F1 Score).

--------------------
PROPOSTA DO SOFTWARE
--------------------

-Proposta fornecida pelo Hemerson Pistori, professor da disciplina de inteligência artificial do curso de Engenharia de computação 8° semestre:  

> Implementar um protótipo de sistema de IA usando aprendizagem automática para realizar análise de sentimento em textos de alguma rede social ou banco anotado disponível online.

-----------------------------
INSTALAÇÃO MANUAL BIBLIOTECAS
-----------------------------

# Caso deseje instalar todas pelo scrip (com um único clique) ou já possua essas bibliotecas instaladas, pule este tópico.

# Com o terminal aberto , insira as bibliotecas abaixo, executando uma de cada vez:

- sudo apt install python3-pip
- pip3 install --user pandas
- pip3 install --user scikit-learn
- pip3 install --user matplotlib
- pip3 install --user numpy
- pip3 install --user googletrans
- sudo apt install python3-tk

-----------------
UTILIZAÇÃO UBUNTU
-----------------

# Habilitar permissão de execução do .sh (abra o terminal no diretório onde se encontra o Analisador_Sentimento.sh e execute as linhas de comandos abaixo, uma de cada vez)

chmod u+x Analisador_Sentimento.sh
chmod a+x Analisador_Sentimento.sh

# Passos

1. Inicialmente realize o download do arquivo.
2. Descompactar o arquivo baixado.
3. Com o terminal aberto e no caminho em que realizou o download do arquivo insira o comando "./Analisador_Sentimento.sh"
4. Será apresentado um menu com três opções, primeiramente deve-se acessar a primeira opção: "instalar_bibliotecas".
5. Após isso será apresentado uma tela na qual constará todos os arquivos que estão sendo instalados.
6. Ao final da instalação será apresentado uma pequena tela, clique em "quit". (Caso apresente)
7. Voltará automaticamente ao MENU, escolha a segunda opção: "executar_analisador"
8. Aguarde um momento enquanto a aplicação inicia.
9. Digite sua frase no campo "Forneça a Frase"
10. Clique em "ANALISAR"
11. O Resultado será exibido abaixo do botão "ANALISAR" (Posistivo ou Negativo)

--------------------------------
EXEMPLOS DE FRASES E SUAS SAÍDAS
--------------------------------

********** Saída Positiva **********


O filme está incrivel!!
__________________________________________________________________

Σε όλα αυτά τα χρόνια, δεν έχω δει ποτέ κάτι τόσο όμορφο και τόσο καλά.
(Em todos esses anos, nunca vi nada tão bonito e tão bom.) - Tradução do Grego
__________________________________________________________________

Еда была превосходной, и обслуживание отличное
(A comida foi excelente e o serviço é ótimo) - Tradução do Russo
__________________________________________________________________

가족과 함께 볼 수있는 좋은 영화
(Bom filme para assistir com a família) - Tradução do Coreano


********** SAÍDA NEGATIVA **********


A comida não estava boa!
__________________________________________________________________

悪い経験でした
(foi uma péssima experiência) - Tradução do Japonês
__________________________________________________________________

c'était une mauvaise expérience
(foi uma experiência ruim) - Tradução do Francês
__________________________________________________________________

De service is verschrikkelijk en het eten is slecht
(O serviço é terrível e a comida é ruim) - Tradução do Holandês
__________________________________________________________________

Szörnyű film, kevés mozgással és kis fejlesztéssel.
(Um filme terrível, com pouco movimento e pouco desenvolvimento.) - Tradução do Húngaro


-----------------------------
SISTEMA OPERACIONAL SUPORTADO
-----------------------------

- Ubuntu 16.04 LTS ou superior

-------------------------------------------------------
REFERÊNCIA PARA CRIAÇÃO DO README E DEPURAÇÃO DO CÓDIGO
-------------------------------------------------------

* https://github.com/thszk/knowledgebase/ 

- Autor: Thiago Suzuque Lodi
- Contribuição: Auxilio na elaboração do README atarvés da disponibilização do modelo presente no link acima, fornecimento de scripts auxiliares para a elaboração da documentação e construção parcial do código, Ajuda na localização e correção de erros. 

----------------------------------------------------------
REFERÊNCIA PARA CRIAÇÃO DO CÓDIGO COM SUAS FUNCIONALIDADES
----------------------------------------------------------


* https://gabrielschade.github.io/2018/04/16/machine-learning-classificador.html

- Autor: Gabriel Schade
- Contribuição: Auxilio no entendimento sobre os fragmentos dos códigos utilizados, esclarecimento sobre funções e bibliotecas tais como: pandas, scikit-learn, matplotlib, numpy, googletran.

* https://github.com/gabrielschade/IA/tree/master/ClassificacaoComentariosComNaiveBayes

- Autor: Gabriel Schade
- Contribuição: Disponibilização do código, funções, importações, instruções de acesso as bancos de frases utilizadas.

* https://docs.python.org/3/library/tkinter.html

- Contribuição: Auxilio na elaboração da interface gráfica, como funções e importações necessárias.

--------------------------------------------
REFERÊNCIA DOS CONJUNTOS DE DADOS UTILIZADOS
--------------------------------------------

# Assuntos dos dados utilizados:

> yelp_labelled.txt - Avaliação de estabelecimentos comerciais alimentícios.
> amazon_cells_labelled - Comentários sobre produtos eletrônicos e outros.
> imdb_labelled.txt - Comentários sobre filmes

# Links:

* https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
* https://www.amazon.com/
* https://www.yelp.com/
* https://www.imdb.com/

