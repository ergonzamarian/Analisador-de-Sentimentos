#!/bin/bash

clear
echo "=================================="                  
echo "    MENU DE EXECUCAO E INSTALCAO	 "
echo "=================================="
select i in instalar_bibliotecas executar_analisador sair
do

	case "$i" in

		instalar_bibliotecas)
			clear
			echo "Instalando bibliotecas..."
			sudo apt install python3-pip
			# python -m ensurepip --upgrade
			pip3 install --user pandas
			pip3 install --user scikit-learn
			pip3 install --user matplotlib
			pip3 install --user numpy
			pip3 install --user googletrans
			sudo apt install python3-tk
			clear
			echo "Bibliotecas instaladas com sucesso, execute o código!!"
			echo "              "
			echo "=================================="                  
			echo "    MENU DE EXECUCAO E INSTALCAO	"
			echo "=================================="
			echo "1) instalar_bibliotecas       	" 
			echo "2) executar_analisador         	"
			echo "3) sair                        	"

			;;

		executar_analisador)
			clear
			echo "Executando - Aguarde um momento..."
			python3 analisador_sentimento.py
			clear
			echo "=================================="                  
			echo "    MENU DE EXECUCAO E INSTALCAO	"
			echo "=================================="
			echo "1) instalar_bibliotecas       	" 
			echo "2) executar_analisador         	"
			echo "3) sair                        	"

			;;

		sair)
			clear
			break
			;;

		*)
			echo "opcao invalida, tente de novo"
			;;
		

	esac

done

exit 0

