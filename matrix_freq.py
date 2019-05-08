from sys import argv
from nphones import nPhones

from collections import defaultdict, Counter
from itertools import chain
import numpy as np

file = open(argv[1], 'r')
phones = nPhones(file.read(),nphone_siz=3)
phones.get_phones()
#print(phones.voc)


#Funcion que crea un vocabulario de palabras con un indice numerico
def vocab():
    vocab = defaultdict()
    vocab.default_factory = lambda: len(vocab)
    return vocab    

#Funcion que pasa la cadena de simbolos a una secuencia con indices numericos
def text2numba(corpus, vocab):
	for doc in corpus:
		yield [vocab[w] for w in doc]

#Obtiene los diccionarios
voc = vocab()
idxphones = list(text2numba(phones.word_phones, voc))

#Obtiene bigramas de nphones
bigrams = [list(zip(cad,cad[1:])) for cad in idxphones] #phones.word_phones
#Obtiene las frecuencias de los bigramas
freqs = Counter(list(chain(*bigrams)))

#Tamaño del vocabulario
N = len(voc)
#Se crea una matriz de unos, para que al aplicar los log se hagan 0 los que nunca pasan
P = np.ones((N,N))

#Frecuencia de unigramas
uni_freqs = Counter(list(chain(*idxphones)))

#Se llena la matriz con las probabilidades
for v,f in freqs.items():
	P[v] = f/(uni_freqs[v[0]])

#El total de tokens (patrticion)
z = sum(uni_freqs.values())

#Se obtiene el log de las probabilidades
logP = np.log(P)

#Se inicializa la entropía en 0
H = 0.0
Hnorm = 0.0
for w,i in voc.items():
	# \sum_j p_ij*logN p_ij
	condHnorm = np.dot(P[i],logP[i])/np.log(N)
	condH = np.dot(P[i],logP[i])

	#\sum_i mu_i \sum_j p_ij*logN p_ij
	Hnorm -= (uni_freqs[i]/z)*condHnorm
	H -= (uni_freqs[i]/z)*condH

print('Entropía:', H)
print('Entropía norm:', Hnorm)
