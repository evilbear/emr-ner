#encoding:utf-8
import chardet, os, codecs, gensim

train_corpus = os.path.dirname(os.path.abspath(__file__)) + '/data.txt'
saved_path = os.path.dirname(os.path.abspath(__file__)) + '/model/model'
# train_corpus = os.path.dirname(os.path.abspath(__file__)) + '/data_split.txt'
# saved_path = os.path.dirname(os.path.abspath(__file__)) + '/model_split/model'
docs = gensim.models.doc2vec.TaggedLineDocument(train_corpus)
model = gensim.models.Doc2Vec(docs, vector_size=600, window=15, min_count=1, sample=1e-5, workers=4, negative=5, dbow_words=1, dm_concat=1)
model.train(docs, total_examples=model.corpus_count, epochs=100)
model.save(saved_path)