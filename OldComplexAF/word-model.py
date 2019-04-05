from __future__ import division
from itertools import chain, combinations
import numpy as np
from collections import Counter
from sys import argv

#Abre el archivo que se va a analizar y separa por lineas
file = open(argv[1],'r').read().strip().split('\n')

#Colecta los tokens del corpus separados por morfemas
words = []
for f in file:
	for w in f.split():
		words.append( w.split('-') )

#Se obtiene la frecuencia del vocabulario (cada uno de los tipos)
W = list(chain(*words))
frVoc = Counter(W)

#Define la duncion para el conteo conjunto de los morfos
def count_conj(lista):
	conj = []
	for word in lista:
	#Toma en cuenta unicamente las palabras que tienen mas de un morfo
		if len(word) > 1:
		#Se obtienen las sucesiones de cadenas de forma estocastica
			for i in range(len(word)-1):
				conj.append ((word[i], word[i+1]) )
	return Counter(list(set(conj)))

#Se obtiene el diccionario de las frecuencias conjuntas
frConj = count_conj(words)

#Se crean identificadores numericos para acceder a las entradas en la matriz
ids = {k:w for k,w in enumerate( frVoc.keys() )}

#Dimension de la matriz
N = len(ids)

#Define una funcion de probabilidad de Lidstone con parametro l
def prob(fr,div,l=0.01):
	return (fr+l)/(div+l*N)

#Se define la matriz de transicion con las probs condicionales
M = np.zeros((N,N))
for i,j in combinations(ids.keys(),2):
	#Define las palabras por el id
	w1 = ids[i]
	w2 = ids[j]
	#Se obtiene la probabilidad condicional y se llena la matriz
	M[i,j] = prob(frConj[(w1,w2)],frVoc[w1])
	M[j,i] = prob(frConj[(w2,w1)],frVoc[w2])


#Obtiene el vector de probabilidades estacionarias Puni
t = sum(frVoc.values())
Puni = np.zeros(N)
for i in ids.keys():
	Puni[i] = prob(frVoc[ids[i]],t)

#Se define la funcion para obtener la entropia del modelo
def H(Arr):
	Id = np.diag(np.ones(N))
	X = -np.log(Arr + Id)/np.log(N)

	#Obtiene la ezperanza condicional de sobre X
	H_X = (Arr*X).sum(1)

	#Calcula las entropias empirica y teorica
	H_e = H_X.sum(0)/N
	H_t = np.dot(Puni,H_X)
	return H_e



print 'Entropia:',H(M)
print 'Perplejidad:', N**H(M)
