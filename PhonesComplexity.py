import numpy as np
from collections import Counter, defaultdict
from re import sub
from itertools import chain, combinations
import numpy as np

class nPhones:
	#file <- Archivo de entrada
	#nphone_siz <- Tamaño de los n-phones/grams que se obtendrán. Si no hay argumento, n=3
	def __init__(self,file,nphone_siz=3):
		super().__init__()
		#Se eliminan signos del archivo de entrada
		self.file = sub(r'[^\w\s]','',file.read().strip().lower()).split()
		self.ngram_siz = nphone_siz

	#Extractror de ngramas
	def ngramas(self,string):
		ngrams = []
		#Se cehca la longitud de la cadena de entrada
		if len(string) < self.ngram_siz:
			ngrams.append(string)
		#Se extraen los ngramas
		else:
			i = 0
			while i + self.ngram_siz - 1 < len(string):
				ngrams.append(string[i:i + self.ngram_siz])
				i += 1
		#Return una lista de ngramas
		return(ngrams)

	#Se obtienen nphones (ngramas a nivel palabra)
	def n_phones(self):
		word_phones = []
		for w in self.file:
			word_phones.append(self.ngramas(w))
		#Regresa una lista de los nphones de cada palabra
		return(word_phones)

	#Define un vocabulario que indexa los nphones {idx:nphone}
	def vocab(self):
		vocab = defaultdict()
		vocab.default_factory = lambda: len(vocab)
		return(vocab)

	#Sustituye los nphones por sus índices númericos
	def word_idx(self,corpus, vocab):
		for doc in corpus:
			yield([vocab[w] for w in doc])

	#Cadena de nphones
	word_phones = None
	#Cadenas de índices numéricos de nphones
	idx_phones = None
	#Vocabulario índice:nphone
	voc = None

	#Obtiene los nphones, el vocabulario, y las cadenas con índices
	def get_phones(self):
		self.word_phones = self.n_phones()
		self.voc = self.vocab()
		self.idx_phones = list(self.word_idx(self.word_phones, self.voc))


class entropy_complexity:
	def __init__(self,nphones):
		super().__init__()
		nphones.get_phones()
		self.Voc = nphones.voc
		self.N = len(self.Voc)
		self.ngrams = nphones.word_phones
		self.size = nphones.ngram_siz

	#Define una funcion de probabilidad
	def get_probs(self):
		frecs = Counter(chain(*self.ngrams))
		last = defaultdict(list)

		#Se ordenan los nphones a partir de la última letra
		#esta letra corresponde a lo que se condicionea en la prob p(last|...)
		for phone in self.Voc.keys():
			last[phone[-1]].append( phone )

		probs = []
		for char, phones in last.items():
			fr_last = np.zeros(len(phones))
			for k,ph in enumerate(phones):
				#frecuencias del nphone fr{p1,p2,...,pi-1,pi}
				fr_last[k] = (frecs[ph])

			#se obtienes las probabilidades fr/sum(fr) para cada nphone
			pre_probs = zip( phones,fr_last/sum(fr_last) )
			probs.append(pre_probs)

		return( dict(chain(*probs)) )
	
	Entropy = None
	#Función para obtener entropia por palabras y el promedio de estas entropias
	def word_entropy(self):
		phone_probs = self.get_probs()
		word_entropies = {}
		
		#Se calcula el promedio por palabra en base a sus nphones
		for w in self.ngrams:
			word = ''.join([p[0] for p in w] + [w[-1][1:]])
			word_entropies[word] = -sum([phone_probs[p]*np.log2(phone_probs[p]) for p in w]) 
	
		#Se obtiene el promedio de estas entropias
		self.Entropy  = sum(word_entropies.values())/len(word_entropies)
		return word_entropies
