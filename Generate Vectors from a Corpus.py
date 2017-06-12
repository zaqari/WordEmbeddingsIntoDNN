import gensim
import codecs
from gensim import corpora, models, similarities
import nltk

#A brief note, and then establishing which corpus you'll be using to generate
# your word-embedding vectors from.
print('For analyzing a vector output of a given word in a corpus use \'model.wv[word]\'')
corpus = input('Where is the corpus you are using?  ')


#An initially empty list that will be filled with the tokens from your corpus.
# 'doc' calls the actual corpus file and then opens it in order to feed through
# the rest of the script. The variable 'readable' simply edits out any residual
# characters that UTF-8 can't interpret, and then the script appends everything
# to the tok_corpus list as a complete sentence containing the entire context in
# your standard CBOW framework.
tok_corpus =[]
doc = codecs.open(corpus, 'r', 'utf-8')
searchlines = doc.readlines()
doc.close()
for i, line in enumerate(searchlines):
    readable = line.replace('\\', '').replace('}', '').replace('uc0u8232', '').replace('\'92', '\'').replace('a0', '').replace('\'93', '\"').replace('\'94', '\"').replace('\'96', ',').replace('\'97', ',').replace('f0fs24 ', '').replace('cf0 ', '').replace('< ', '').replace(' >', '').replace('\r\n', '')
    tok_corpus.append(nltk.word_tokenize(readable))


#These last two, thus, generate your model and simultanesouly saves it once the
# system has finished rendering word-embeddings for all the content in the
# corpus.
model = gensim.models.Word2Vec(tok_corpus, min_count=2, size=100)
model.save(input('Where would you like to save your model?  '))


