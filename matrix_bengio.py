from sys import argv
from nphones import nPhones
from Neural import Model
import numpy as np

file = open(argv[1], 'r')
phones = nPhones(file.read(),nphone_siz=1)
phones.get_phones()

print(phones.voc)

model = Model(phones.word_phones, ngramas=2)
model.train(its=50)

N = len(phones.voc)
P = np.zeros((N,N))
for w in model.voc:
#	print(phones.voc[w])
	P[phones.voc[w]] = model.forward([w])

k = P.sum(0)
mu = k/k.sum(0)

logP = np.log(P)

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


'''
EOS = '<EOS>'
BOS = '<BOS>'

corpus1 = ['el perro come un hueso', 'un perro come un hueso', 'el nino salta', 'un nino come','el perro come un hueso', 'un perro come un hueso', 'el nino salta', 'un nino come']

cadenas = [[BOS] + cad.split() + [EOS] for cad in corpus1]
model = Model(cadenas, ngramas=2)
model.train()
print(model.voc)
print(model.forward(['<BOS>'])[model.voc['un']])
'''

#print(phones.word_phones)
#print(model.forward(['per']))
