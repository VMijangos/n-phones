# n-phones

Se tienen dos scripts principales:

1) matrix_freq.py - Calcula la entropía en las palabras, basada en los n-phones, con base en un modelo del lenguaje basado en una probabilidad frecuentista.

2) matrix_bengio.py - Calcula la entropía en las palabras, basada en los n-phones, con base en un modelo del lenguaje basado en la propuesta de Bengio (2003).

Otros scripts auxiliares son:

a) nphones.py - Clase para extraer n-phones, creando un vocabulario indexado.
b) Neural.py - Contiene el modelo del lenguaje neuronal de Bengio (2003)+

Forma de correr:

>> python3 matrix_freq.py corpus/nombre_de_corpus.txt
