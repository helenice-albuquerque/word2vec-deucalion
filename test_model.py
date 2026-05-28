from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_final.model")

# Palavras similares
print(model.wv.most_similar("cancer"))
print(model.wv.most_similar("heart"))
