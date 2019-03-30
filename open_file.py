from sys import argv
import PhonesComplexity as cplx

file = open(argv[1], 'r')
phones = cplx.nPhones(file,nphone_siz=3)
words = phones.file
H = cplx.entropy_complexity(phones)
word_entropies = H.word_entropy()

for w in words:
	print(w, word_entropies[w])

print('Avg. entropy:', H.Entropy)
