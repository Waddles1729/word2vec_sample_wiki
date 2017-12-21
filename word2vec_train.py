from gensim.models import word2vec
import logging
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = codecs.open('jawiki_wakati.txt')
sentences = word2vec.Text8Corpus('jawiki_wakati.txt')

model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)

model.save("jawiki_wakati.model")
