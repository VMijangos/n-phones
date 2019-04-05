from __future__ import division
import numpy as np
from sys import argv
from gensim.models.word2vec import *

#Abre el archivo que se va a analizar
file = open( argv[1], 'r').read().strip().split('\n')

#Colecta los tokens del corpus separados por morfemas
words = []
for f in file:
	for w in f.split():
		words.append( w.split('-') )

#Genera el modelo de Word2Vec
l = 100
model = Word2Vec(words, size=l, window=5, min_count=0, workers=4)

#Genera una matriz del tamano del vocabulario por el num de dimensiones
Voc = {i:w for w,i in enumerate(model.vocab.keys())}
N = len(Voc)
A = np.zeros((N,l))

#Llena la matriz con los vectores de palabra
for w,i in Voc.iteritems():
	A[i] = model[w]

#Determina la probabilidad Softmax
P = np.exp( np.dot(A, A.T))
M = P*(1/P.sum(0))

#Define la funcion para calcular perplejidad
def H(Arr):
	X = -np.log(Arr)/np.log(N)

	#Obtiene la ezperanza condicional de sobre X
	H_X = (Arr*X).sum(1)

	#Calcula las entropias empirica y teorica
	H_e = H_X.sum(0)/N
	#H_t = np.dot(Puni,H_X)
	return H_e



print 'Entropia:',H(M)
print 'Perplejidad:', N**H(M)
