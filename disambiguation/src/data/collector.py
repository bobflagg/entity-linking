import codecs
from domain import Entity, Mention
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
import time

def collect(home, directory, target, update):
  print "Collecting mentions data..."
  start = time.time()
  path = "%s/%s/%s-mention" % (home, directory, target)
  if update or not os.path.exists(path):
    data = codecs.open("%s/%s/%s-data" % (home, directory, target), 'r', 'UTF-8')
    doc_id = None
    mention = None
    finished_doc = True
    fpm = codecs.open(path, 'w', 'utf-8')
    fpc = codecs.open("%s/%s/%s-context" % (home, directory, target), 'w', 'utf-8')
    fpl = codecs.open("%s/%s/%s-info" % (home, directory, target), 'w', 'utf-8')
    index = 0
    for line in data: 
      line = line.strip()
      if line:
        if finished_doc: 
          doc_id, path = line.split(":")
          doc_id = doc_id.strip()
          path = path.strip()[1:-1]
          finished_doc = False
        else:
          data = line.split('\t')
          mention = Mention(doc_id, data[0], data[-1])
          mention.add_featureSets(data[1:-1])
          store_mention_data(mention, index, fpc, fpl, fpm)
          index += 1
      else: 
        finished_doc = True
    fpm.close()
    fpc.close()
    fpl.close()
    fp = codecs.open("%s/%s/%s-entity" % (home, directory, target), 'w', 'utf-8')
    keys = Entity.MAP.keys()  
    for key in sorted(keys, key=lambda x: Entity.MAP[x].index):
      entity = Entity.MAP[key]
      fp.write("%s\t%s\n" % (entity.subtype, entity.phrase))
    fp.close()
    fp = codecs.open("%s/%s/%s-shape" % (home, directory, target), 'w', 'utf-8')
    fp.write("%d\t%d\n" % (index, Entity.NEXT_INDEX))
    fp.close()
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)

def store_mention_data(mention, index, fpc, fpl, fpm):
  fpm.write("%d\t%s\n" % (index, mention.record()))
  fpc.write("%d\t%s\n" % (index, mention.context))
  fpl.write("%d\t%s\n" % (index, mention.info()))

def get_features(home, directory, target):
  fp = codecs.open("%s/%s/%s-mention" % (home, directory, target), 'r', 'utf-8')
  for record in fp:
    parts = record.strip().split("\t")
    i = int(parts[0])
    for item in parts[1:]:
      j, value = item.split(":")
      j = int(j)
      value = float(value)
      yield i, j, value
  fp.close()

def get_info(home, directory, target):
  fp = codecs.open("%s/%s/%s-info" % (home, directory, target), 'r', 'utf-8')
  for record in fp:
    parts = record.strip().split("\t")
    yield int(parts[0]), parts[3]
  fp.close()

def get_sparse_matrix(home, directory, target, update=False):
  print "Getting sparse matrix..."
  start = time.time()
  path = "%s/%s/%s-sparse.npz" % (home, directory, target)
  nrow, n_entities = get_shape(home, directory, target)
  ncol = 2 * n_entities
  if update or not os.path.exists(path):
    S = lil_matrix((nrow, ncol))
    print '1', S.shape
    for i, j, value in get_features(home, directory, target): S[i,j] = value
    #for i, j, value in features: S[i,j] = value
    S = S.tocsc()
    print '2', S.shape
    np.savez(path, S.data, S.indices, S.indptr)
  else:
    npzfile = np.load(path)
    S = csc_matrix((npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']), shape=(nrow, ncol))
    print '3', S.shape
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return S

def get_reduced_matrix(X, components, use_tf_idf, update=False):
  print "Getting reduced matrix..."
  start = time.time()
  path = "%s/%s/%s-reduced.npz" % (home, directory, target)
  if update or not os.path.exists(path):
    if use_tf_idf: X = tf_idf(X)
    lsa = TruncatedSVD(components)
    X = lsa.fit_transform(X)
    X = Normalizer(copy=False).fit_transform(X)
    np.savez(path, X)
  else:
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

def get_shape(home, directory, target):
  fp = codecs.open("%s/%s/%s-shape" % (home, directory, target), 'r', 'utf-8')
  n_mentions, n_entities = fp.readline().split("\t")
  fp.close()
  return int(n_mentions), int(n_entities)

def get_indices(home, directory, target, surface_form):
  return [i for i, phrase in get_info(home, directory, target) if surface_form in phrase]
    
home = "/home/disambiguation/"
directory = "data-sets"
target = "js-test"
update_mention_data = False
update_reduced_matrix = False
update_sparse_matrix = False
components = 80
use_tf_idf = True
surface_form = "Smith"

if __name__ == "__main__":
  print "Clustering data..."
  start = time.time()
  collect(home, directory, target, update_mention_data)
  S = get_sparse_matrix(home, directory, target, update_sparse_matrix)
  X = get_reduced_matrix(S, components, use_tf_idf, update_reduced_matrix)
  print X.shape
  info = [(i, phrase) for i, phrase in get_info(home, directory, target)]
  indices = [i for i, phrase in info if surface_form in phrase]
  data = X[indices,:]
  R = dendrogram(linkage(data, method='complete'), orientation="left")
  finish = time.time()
  print 'Total time: %0.3f s' % (finish-start)
  plt.ylabel('Mentions')
  plt.xlabel('Height')
  plt.suptitle('Cluster Dendrogram', fontweight='bold', fontsize=14);
  plt.show()
