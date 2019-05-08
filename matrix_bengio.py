from sys import argv
from nphones import nPhones
from Neural import Model
import numpy as np

#abre el archivo
file = open(argv[1], 'r')

#extrae los nphones, variar este parámetro implica tomar uniphones, triphones u otros
phones = nPhones(file.read(),nphone_siz=1)
phones.get_phones()

#Aprende el modelo de Bengio
model = Model(phones.word_phones, ngramas=2)
model.train(its=50)

#Tamaño de nphones
N = len(phones.voc)

#Crea matriz de probabilidades de transición
P = np.zeros((N,N))
for w in model.voc:
#	print(phones.voc[w])
	P[phones.voc[w]] = model.forward([w])

#Probabilidades marginales a partir de P
k = P.sum(0)
mu = k/k.sum(0)

#Crea una matriz con log(p_ij)
logP = np.log(P)

#Calcula las entropias: normal (H) y normalizada entre 0 y 1 (Hnorm)
Hnorm = 0.0
H = 0.0
for w,i in phones.voc.items():
	# \sum_j p_ij*logN p_ij
	condHnorm = np.dot(P[i],logP[i])/np.log(N)
	condH = np.dot(P[i],logP[i])

	#\sum_i mu_i \sum_j p_ij*logN p_ij
	Hnorm -= (mu[i])*condHnorm
	H -= (mu[i])*condH

print('Entropía:', H)
print('Entropía norm:', Hnorm)
