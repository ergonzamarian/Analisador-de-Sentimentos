#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

# NOME: ERGON ZAMARIAN LIMA
# RA: 167776

# ANALISADOR - FONTES DE PESQUISA
# https://gabrielschade.github.io/2018/04/16/machine-learning-classificador.html
# https://github.com/gabrielschade/IA/tree/master/ClassificacaoComentariosComNaiveBayes

# py ensurepip
# py -m ensurepip --upgrade
# py -m pip install --user pandas
# py -m pip install --user scikit-learn
# py -m pip install --user matplotlib
# py -m pip install --user numpy
# py -m pip install --user googletrans
# py -m tkinter


from googletrans import Translator # biblioteca para traduzir texto
from googletrans.gtoken import TokenAcquirer # biblioteca para traduzir texto
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from tkinter import *# biblioteca interface gráfica

import os
import sqlite3
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas

# lendo e obtendo dados 
def obter_todos_dados():

    # com a estratégia abaixo podemos executar o programa em qualuer diretório desde que ele esteja na mesma pasta do .bat ou.sh
    dn1 = os.path.dirname(os.path.realpath(__file__))
    dir_path1 = os.path.join(dn1,"imdb_labelled.txt")

    dn2 = os.path.dirname(os.path.realpath(__file__))
    dir_path2 = os.path.join(dn2,"amazon_cells_labelled.txt")

    dn3 = os.path.dirname(os.path.realpath(__file__))
    dir_path3 = os.path.join(dn3,"yelp_labelled.txt")

    with open(dir_path1, "r") as texto:
        dados = texto.read().split('\n')
         
    with open(dir_path2, "r") as texto:
        dados += texto.read().split('\n')

    with open(dir_path3, "r") as texto:
        dados += texto.read().split('\n')

    return dados

valores = obter_todos_dados()

def pre_processamento_dados(dados):
    processando_dados = []
    for dados_unicos in dados:
        if len(dados_unicos.split("\t")) == 2 and dados_unicos.split("\t")[1] != "":
            processando_dados.append(dados_unicos.split("\t"))

    return processando_dados

todos_dados = obter_todos_dados()
valores = pre_processamento_dados(todos_dados)
# dividindo dados para treinamento, o conjunto de treinamento foi armazenado em "dados"
def dividir_dados(dados):
    total = len(dados)
    taxa_treinamento = 0.75
    dados_treinamento = []
    dados_avaliacao = []
    
    for indice in range(0, total):
        if indice < total * taxa_treinamento:
            dados_treinamento.append(dados[indice])
        else:
            dados_avaliacao.append(dados[indice])

    return dados_treinamento, dados_avaliacao

def etapa_pre_processamento():
    dados = obter_todos_dados()
    processando_dados = pre_processamento_dados(dados)

    return dividir_dados(processando_dados)

def etapa_treinamento(dados, vectorizer):
    texto_treinamento = [dados[0] for dados in dados]
    resultado_treinamento = [dados[1] for dados in dados]

    texto_treinamento = vectorizer.fit_transform(texto_treinamento)

    return BernoulliNB().fit(texto_treinamento, resultado_treinamento)

dados_treinamento, dados_avaliacao = etapa_pre_processamento()
vectorizer = CountVectorizer(binary = 'true')
classificador = etapa_treinamento(dados_treinamento, vectorizer)

def analisar_texto(classificador, vectorizer, texto):
    return texto, classificador.predict(vectorizer.transform([texto]))

def print_resultado(resultado):
    texto, resultado_analise = resultado
    neg_pos = "Positivo" if resultado_analise[0] == '1' else "Negativo"
    return neg_pos

def avaliacao_simples(dados_avaliacao):
    texto_avaliacao     = [dados_avaliacao[0] for dados_avaliacao in dados_avaliacao]
    resultado_avaliacao   = [dados_avaliacao[1] for dados_avaliacao in dados_avaliacao]

    total = len(texto_avaliacao)
    correto = 0
    for indice in range(0, total):
        resultado_analise = analisar_texto(classificador, vectorizer, texto_avaliacao[indice])
        texto, resultado = resultado_analise
        correto += 1 if resultado[0] == resultado_avaliacao[indice] else 0

    return correto * 100 / total

avaliacao_simples(dados_avaliacao)
def sair():
            sys.exit()
            
# criando confusion_matrix a fim de utiliza-la para os cálculos de acuracia, precisão recall e F1 Score
def criar_confusion_matrix(dados_avaliacao):
    texto_avaliacao      = [dados_avaliacao[0] for dados_avaliacao in dados_avaliacao]
    resultado_atual      = [dados_avaliacao[1] for dados_avaliacao in dados_avaliacao]
    resultado_previsao   = []
    for texto in texto_avaliacao:
        resultado_analise = analisar_texto(classificador, vectorizer, texto)
        resultado_previsao.append(resultado_analise[1][0])
    
    matrix = confusion_matrix(resultado_atual, resultado_previsao)
    return matrix
    
confusion_matrix_resultado = criar_confusion_matrix(dados_avaliacao)
verdadeiros_negativos = confusion_matrix_resultado[0][0]
falsos_negativos = confusion_matrix_resultado[0][1]
falsos_positivos = confusion_matrix_resultado[1][0]
verdadeiros_positivos = confusion_matrix_resultado[1][1]

acuracia = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falsos_positivos + falsos_negativos)
precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
Retorno = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
point_f1 = 2*(Retorno * precisao) / (Retorno + precisao)

# INTERFACE
class Application:
    def __init__(self, master=None):
        
        self.fontePadrao = ("Arial", "10")
        self.primeiroContainer = Frame(master)
        self.primeiroContainer["pady"] = 30
        self.primeiroContainer.pack()
  
        self.segundoContainer = Frame(master)
        self.segundoContainer["padx"] = 20
        self.segundoContainer.pack()

        self.terceiroContainer = Frame(master)
        self.terceiroContainer["pady"] = 20
        self.terceiroContainer.pack()

        # acuracia
        self.quartoContainer = Frame(master)
        self.quartoContainer["pady"] = 20
        self.quartoContainer.pack()
        # precisao
        self.quintoContainer = Frame(master)
        self.quintoContainer["pady"] = 20
        self.quintoContainer.pack()
        # Recall
        self.sextoContainer = Frame(master)
        self.sextoContainer["pady"] = 20
        self.sextoContainer.pack()
        # F1 Score
        self.setimoContainer = Frame(master)
        self.setimoContainer["pady"] = 20
        self.setimoContainer.pack()
        
        self.titulo = Label(self.primeiroContainer, text="ANALISADOR DE SENTIMENTOS (Multi-Language)")
        self.titulo["font"] = ("Arial", "10", "bold")
        self.titulo.pack()
    
        self.FraseLabel = Label(self.segundoContainer,text="Forneça a Frase ", font=self.fontePadrao)
        self.FraseLabel.pack(side=LEFT)
  
        self.frase = Entry(self.segundoContainer)
        self.frase["width"] = 100
        self.frase["font"] = self.fontePadrao
        self.frase.pack(side=LEFT)

        self.analisar = Button(self.terceiroContainer)
        self.analisar["text"] = "ANALISAR"
        self.analisar["font"] = ("Calibri", "10", "bold")
        self.analisar["width"] = 12
        self.analisar["command"] = self.print_analise
        self.analisar.pack()
  
        self.mensagem = Label(self.terceiroContainer, text="", font=self.fontePadrao)
        self.mensagem.pack()

        self.acuraciaL = Label(self.quartoContainer, text="Acurácia (Avaliação): "+str(acuracia))
        self.acuraciaL["font"] = ("Arial", "10", "bold")
        self.acuraciaL.pack(side=LEFT)

        self.precisaoL = Label(self.quintoContainer, text=" Precisão: "+str(precisao))
        self.precisaoL["font"] = ("Arial", "10", "bold")
        self.precisaoL.pack(side=LEFT)

        self.retornoL = Label(self.sextoContainer, text="Recall: "+str(Retorno))
        self.retornoL["font"] = ("Arial", "10", "bold")
        self.retornoL.pack(side=LEFT)

        self.F1_L = Label(self.setimoContainer, text=" F1 Score: "+str(point_f1))
        self.F1_L["font"] = ("Arial", "10", "bold")
        self.F1_L.pack(side=LEFT)

    # Analisando Frase
    def print_analise(self):
        
        # IDENTIFICANDO E TRADUZINDO IDIOMA
        frase_use = self.frase.get() 

        translator = Translator()
        idioma = translator.detect(frase_use)
        acquirer = TokenAcquirer()
        acquirer.do(frase_use)
        frase_traduzida = frase_use
        frase_traduzida = translator.translate(frase_traduzida, src = idioma.lang, dest = "en")

        texto = 'POSITIVO'

        if  print_resultado(analisar_texto(classificador, vectorizer,frase_traduzida.text)) == "Positivo":
            
            self.mensagem["text"] = texto
        else:
           
            texto = "NEGATIVO"
            self.mensagem["text"] = texto
    

root = Tk()
Application(root)
root.mainloop()
