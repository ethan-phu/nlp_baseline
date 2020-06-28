from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
path = os.path.abspath('.')
glove_file = datapath(path+'/embeddings/result.txt')
# tmp_file = get_tmpfile(path+"/embeddings/to_word2vec.txt")
# glove2word2vec(glove_file, tmp_file)
wvmodel = KeyedVectors.load_word2vec_format(glove_file)

print (wvmodel.most_similar("apple"))