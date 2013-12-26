'''
Various methods for preprocessing mentions data in preparation for clustering and disambiguation.
'''
import codecs
import ConfigParser
from data.corpus_builder import DirectoryBackedCorpus, FileBackedCorpus
from domain import Entity, Mention
from gensim import corpora, models
from kpe.key_concept_extractor import extract_kc
import matplotlib.pyplot as plt
import numpy as np
import optparse
import os
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

OUTPUT_DIRECTORY = "output"

def collect(home, directory, target, output_directory, update):
  '''
  Collects and stores entities and mentions together with count and
  proximity features for each mention.
  '''
  start = time.time()
  path = "%s/mention" % output_directory
  if update or not os.path.exists(path):
    print "Collecting mentions data..."
    data = codecs.open("%s/%s/%s-data" % (home, directory, target), 'r', 'UTF-8')
    doc_id = None
    mention = None
    finished_doc = True
    fpm = codecs.open(path, 'w', 'utf-8')
    fpc = codecs.open("%s/context" % output_directory, 'w', 'utf-8')
    fpl = codecs.open("%s/info" % output_directory, 'w', 'utf-8')
    index = 0
    for line in data: 
      line = line.strip()
      if line:
        if finished_doc: 
          doc_id, path = line.split(":")
          doc_id = doc_id.strip()
          path = path.strip()[1:-1]
          doc_name = path.split('/')[-1]
          finished_doc = False
        else:
          data = line.split('\t')
          mention = Mention(doc_id, doc_name, data[0], data[-1])
          mention.add_featureSets(data[1:-1])
          store_mention_data(mention, index, fpc, fpl, fpm)
          index += 1
      else: 
        finished_doc = True
    fpm.close()
    fpc.close()
    fpl.close()
    fp = codecs.open("%s/entity" % output_directory, 'w', 'utf-8')
    keys = Entity.MAP.keys()  
    for key in sorted(keys, key=lambda x: Entity.MAP[x].index):
      entity = Entity.MAP[key]
      fp.write("%s\t%s\n" % (entity.subtype, entity.phrase))
    fp.close()
    fp = codecs.open("%s/shape" % output_directory, 'w', 'utf-8')
    fp.write("%d\t%d\n" % (index, Entity.NEXT_INDEX))
    fp.close()
  else: print "Mentions data appears to be up-to-date."
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)

def get_output_directory(home, target):
  path = "%s/%s/%s" % (home, OUTPUT_DIRECTORY, target)
  if not os.path.exists(path): os.makedirs(path) 
  return path

def store_mention_data(mention, index, fpc, fpl, fpm):
  fpm.write("%d\t%s\n" % (index, mention.record()))
  fpc.write("%d\t%s\n" % (index, mention.context))
  fpl.write("%d\t%s\n" % (index, mention.info()))

def get_features(output_directory):
  '''
  Generates entity count and entity proximity features for all mentions.
  '''
  _, ncol = get_shape(output_directory)
  fp = codecs.open("%s/mention" % output_directory, 'r', 'utf-8')
  for record in fp:
    parts = record.strip().split("\t")
    i = int(parts[0])
    for item in parts[1:]:
      t, j, value = item.split(":")
      j = int(j)
      if t == 'p': j = ncol  + int(j)
      value = float(value)
      yield i, j, value
  fp.close()

def get_info(output_directory):
  fp = codecs.open("%s/info" % output_directory, 'r', 'utf-8')
  current_doc = None
  doc_index = 0
  for record in fp:
    parts = record.strip().split("\t")
    doc = parts[1]
    if current_doc and not current_doc == doc: doc_index += 1
    current_doc = doc
    yield (int(parts[0]), doc_index, parts[3])
  fp.close()

def extract_context_features(output_directory, target, no_below=2, no_above=0.5, update=False):
  print "Extracting context features..."
  start = time.time()
  path = '/%s/context-tfidf.mm' % output_directory
  if update or not os.path.exists(path):
    context_corpus_path = "%s/context" % output_directory
    description = "%s context corpus" % target
    corpus = FileBackedCorpus(context_corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('/%s/context.dict' % output_directory)
    corpora.MmCorpus.serialize('/%s/context.mm' % output_directory, corpus)
    tfidf = models.TfidfModel(corpus)
    tfidf.save('/%s/context.tfidf' % output_directory)
    corpus_tfidf = tfidf[corpus]
    corpora.MmCorpus.serialize(path , corpus_tfidf)
  dictionary = corpora.Dictionary.load('/%s/context.dict' % output_directory)
  tfidf = models.TfidfModel.load('/%s/context.tfidf' % output_directory)
  corpus_tfidf = corpora.MmCorpus(path)
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return dictionary, tfidf , corpus_tfidf

def extract_keyphrase_features(home, output_directory, target, update=False):
  print "Extracting keyphrase features..."
  start = time.time()
  path = "%s/keyphrase-corpus.mm" % output_directory
  if update or not os.path.exists(path):
    keyphrases = set()
    corpus_path = "%s/data-sets/%s-doc" % (home, target)
    description = "%s corpus" % target
    corpus = DirectoryBackedCorpus(corpus_path, description)
    fpis = codecs.open('/%s/keyphrase-indices' % output_directory, 'w', 'utf-8')
    fpi = codecs.open('/%s/keyphrase-index' % output_directory, 'w', 'utf-8')
    ncol = 0
    keyphrase_index_map = {}
    kp_corpus = []
    for doc in corpus.documents:
      keyphrases = extract_kc(doc.get_text())
      indices = []
      for keyphrase in keyphrases:
        if not keyphrase in keyphrase_index_map:
          keyphrase_index_map[keyphrase] = ncol
          fpi.write("%s\t%d\n" % (keyphrase, ncol))
          ncol += 1
        indices.append(keyphrase_index_map[keyphrase])
      fpis.write("%s\t%s\n" % (doc.get_id(), indices))
      kp_corpus.append([(i, 1) for i in indices])
    fpis.close()
    fpi.close()
    corpora.MmCorpus.serialize(path, kp_corpus)
    nrow = len(kp_corpus)
    #S = lil_matrix((nrow, ncol))
    #for i in range(nrow):
    #  for j in features[i]: S[i,j] = 1
    #S = S.tocsc()
    #np.savez(path, S.data, S.indices, S.indptr)
    fp = codecs.open("%s/keyphrase-shape" % output_directory, 'w', 'utf-8')
    fp.write("%d\t%d\n" % (nrow, ncol))
    fp.close()
  nrow, ncol = get_shape(output_directory, path="keyphrase-shape")
  kp_corpus= corpora.MmCorpus(path)
  #npzfile = np.load(path)
  #S = csc_matrix((npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']), shape=(nrow, ncol))
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return kp_corpus

def extract_topic_features(home, output_directory, target, num_topics, no_below=2, no_above=0.5, update=False):
  print "Extracting topic features..."
  start = time.time()
  path = '/%s/document-lda.mm' % output_directory
  if update or not os.path.exists(path):
    corpus_path = "%s/data-sets/%s-doc" % (home, target)
    description = "%s corpus" % target
    corpus = DirectoryBackedCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('/%s/document.dict' % output_directory)
    corpora.MmCorpus.serialize('/%s/document.mm' % output_directory, corpus)
    lda = models.ldamodel.LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
    lda.save('/%s/document.lda' % output_directory)
    corpus_lda = lda[corpus]
    corpora.MmCorpus.serialize(path, corpus_lda)
  dictionary = corpora.Dictionary.load('/%s/document.dict' % output_directory)
  lda = models.ldamodel.LdaModel.load('/%s/document.lda' % output_directory)
  corpus_lda = corpora.MmCorpus(path)
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return dictionary, lda, corpus_lda

def get_sparse_matrix(output_directory, context_corpus_tfidf, n_context_features, document_corpus_lda, num_topics, keyphrase_corpus, update=False):
  start = time.time()
  path = "%s/sparse.npz" % output_directory
  _, ncol_kf = get_shape(output_directory, path="keyphrase-shape")
  nrow, n_entities = get_shape(output_directory)
  ncol = 2 * n_entities + n_context_features + num_topics + ncol_kf
  if update or not os.path.exists(path):
    print "Updating sparse matrix..."
    S = lil_matrix((nrow, ncol))
    # basic features: entity coreference and proximity
    for i, j, value in get_features(output_directory): S[i,j] = value
    # context features
    i = 0
    for doc in context_corpus_tfidf: 
      for index, value in doc: S[i, 2 * n_entities + index] = value
      i += 1
    # topic and keyphrase features
    for mention_index, doc_index, _ in get_info(output_directory):
      # topic features 
      for index, value in document_corpus_lda[doc_index]: S[mention_index, 2 * n_entities + n_context_features + index] = value
      # keyphrase features
      for index, value in keyphrase_corpus[doc_index]: S[mention_index, 2 * n_entities + n_context_features + num_topics + index] = value
    # keyphrase features
    # si = 1
    # sp = S.indptr[si]
    # col = 0
    # for i in range(len(S.data)):
    #   if i == sp:
    #     col += 1
    #     si += 1
    #     sp = S.indptr[si]
    #   row = S.indices[i]     
    S = S.tocsc()
    np.savez(path, S.data, S.indices, S.indptr)
  else: print "Loading sparse matrix..."
  npzfile = np.load(path)
  S = csc_matrix((npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']), shape=(nrow, ncol))
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return S

def get_reduced_matrix(output_directory, X, components, use_tf_idf, update=False):
  start = time.time()
  path = "%s/reduced.npz" % output_directory
  if update or not os.path.exists(path):
    print "Building and storing reduced matrix..."
    if use_tf_idf: X = tf_idf(X)
    lsa = TruncatedSVD(components)
    X = lsa.fit_transform(X)
    X = Normalizer(copy=False).fit_transform(X)
    np.savez(path, X)
  else: print "Loading reduced matrix..."
  npzfile = np.load(path)
  X = npzfile['arr_0']
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return X

def tf_idf(X):
  tfidfTransformer = TfidfTransformer(use_idf=True)
  tfidfTransformer.fit(X)
  X = tfidfTransformer.transform(X, copy=True)
  return X

def get_shape(output_directory, path="shape"):
  fp = codecs.open("%s/%s" % (output_directory, path), 'r', 'utf-8')
  nrow, ncol = fp.readline().split("\t")
  fp.close()
  return int(nrow), int(ncol )

def get_indices(output_directory, surface_form):
  return [i for i, phrase in get_info(output_directory) if surface_form in phrase]

#  PYTHONPATH=/home/disambiguation/entity-linking/disambiguation/src
if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()
  parser.add_option("-m", help="Update mentions data", dest='update_mentions_data', default=False, action='store_true')
  parser.add_option("-c", help="Update context features", dest='update_context_features', default=False, action='store_true')
  parser.add_option("-t", help="Update topic features", dest='update_topic_features', default=False, action='store_true')
  parser.add_option("-k", help="Update keyphrasefeatures", dest='update_keyphrase_features', default=False, action='store_true')
  parser.add_option("-r", help="Update reduced matrix", dest='update_reduced_matrix', default=False, action='store_true')
  parser.add_option("-s", help="Update sparse matrix", dest='update_sparse_matrix', default=False, action='store_true')
  (opts, args) = parser.parse_args()
  if opts.update_mentions_data or opts.update_context_features or opts.update_topic_features or opts.update_keyphrase_features: opts.update_sparse_matrix = True
  if opts.update_sparse_matrix: opts.update_reduced_matrix = True
  use_tf_idf = True
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/js.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  data_directory = config.get('main', 'data-directory')
  target = config.get('main','target')
  surface_form = config.get('main','surface-form')
  number_of_components = config.getint('dimensionality-reduction','number-of-components')
  # Collect data
  output_directory = get_output_directory(home, target)
  #	- mention
  collect(home, data_directory, target, output_directory, opts.update_mentions_data)
  #	- context
  no_below = config.getint('context-corpus','no-below')
  no_above = config.getfloat('context-corpus','no-above')
  context_dictionary, context_tfidf , context_corpus_tfidf = extract_context_features(output_directory, target, no_below=no_below, no_above=no_above, update=opts.update_context_features)
  #	- topic
  no_below = config.getint('lda','no-below')
  no_above = config.getfloat('lda','no-above')
  num_topics = config.getint('lda','num-topics')
  document_dictionary, document_lda ,document_corpus_lda = extract_topic_features(home, output_directory, target, num_topics, no_below=no_below, no_above=no_above, update=opts.update_topic_features)
  #	- keyphrase
  keyphrase_corpus = extract_keyphrase_features(home, output_directory, target, update=opts.update_keyphrase_features)
  #	- sparse matrix
  S = get_sparse_matrix(output_directory, context_corpus_tfidf, len(context_dictionary), list(document_corpus_lda), num_topics, keyphrase_corpus, update=opts.update_sparse_matrix)
  #	- reduced matrix
  number_of_components = config.getint('dimensionality-reduction','number-of-components')
  X = get_reduced_matrix(output_directory, S, number_of_components, use_tf_idf, update=opts.update_reduced_matrix)
  #	- filtering
  info = [(i, phrase) for i, _, phrase in get_info(output_directory)]
  indices = [i for i, phrase in info if surface_form in phrase]
  data = X[indices,:]
  indices  = np.array(indices)
  np.savez("%s/data.npz" % output_directory, data, indices)
