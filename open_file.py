from sys import argv
from nphones import nPhones

file = open(argv[1], 'r')
phones = nPhones(file)
phones.get_phones()

print(phones.idx_phones)
print(phones.word_phones)
