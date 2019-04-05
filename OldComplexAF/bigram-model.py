from __future__ import division
from itertools import chain, combinations
import numpy as np
from collections import Counter
from sys import argv

#Abre el archivo que se va a analizar y separa por lineas
file = open(argv[1],'r').read().strip().lower().split('\n')

#Colecta los tokens del corpus separados por morfemas
words = []
for f in file:
	if f != '':
		words.append( chain(*[w.split('-') for w in f.split()]) )

words = list(chain(*words))

frWords = Counter(words)
bigrams = Counter( zip(words, words[1:]) )

#Se crean identificadores numericos para acceder a las entradas en la matriz
ids = {k:w for k,w in enumerate( frWords.keys() )}

#Dimension de la matriz
N = len(frWords.keys())

#Define una funcion de probabilidad de Lidstone con parametro l
def prob(fr,div,l=1):
	return (fr+l)/(div+l*(N-1))

#Se define la matriz de transicion con las probs condicionales
M = np.zeros((N,N))
for i,j in combinations(ids.keys(),2):
	#Define las palabras por el id
	w1 = ids[i]
	w2 = ids[j]
	#Se obtiene la probabilidad condicional y se llena la matriz
	M[i,j] = prob(bigrams[(w1,w2)],frWords[w1])
	M[j,i] = prob(bigrams[(w2,w1)],frWords[w2])

#Obtiene el vector de probabilidades estacionarias Puni
t = sum(frWords.values())
Puni = np.zeros(N)
for i in ids.keys():
	Puni[i] = prob(frWords[ids[i]],t)

#Se define la funcion para obtener la entropia del modelo
def H(Arr):
	Id = np.diag(np.ones(N))
	X = -np.log(Arr + Id)/np.log(N)

	#Obtiene la ezperanza condicional de sobre X
	H_X = (Arr*X).sum(1)

	#Calcula las entropias empirica y teorica
	H_e = H_X.sum(0)/N
	H_t = np.dot(Puni,H_X)
	return H_t



print H(M)
print N**H(M)
