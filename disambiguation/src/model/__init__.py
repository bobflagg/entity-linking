from data.corpus_builder import DirectoryBackedCorpus, FileBackedCorpus
from gensim import models
from gensim.matutils import corpus2dense
import os

def build_lda_topic_model(path, description, num_topics, no_below=1, no_above=1.0):
  if os.path.isdir(path): 
    print "building DirectoryBackedCorpus"
    corpus = DirectoryBackedCorpus(path, description)
  else: 
    print "building FileBackedCorpus"
    corpus = FileBackedCorpus(path, description, no_below=2, no_above=1.0)
  print "build_lda_topic_model: initialize_dictionary"
  for doc in corpus.documents:
    print doc.get_tokens()
  for vec in corpus:
    print vec
  corpus.initialize_dictionary()
  print corpus.dictionary
  print corpus2dense(corpus, 12)
  print corpus.dictionary.token2id
  new_doc = "Human computer interaction"
  new_vec = corpus.dictionary.doc2bow(new_doc.lower().split())
  print new_vec # the word "interaction" does not appear in the   print "build_lda_topic_model: LdaModel"
  model = models.ldamodel.LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
  return corpus, model