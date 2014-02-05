import codecs
import ConfigParser
import csv
from data.domain import SUBTYPES
import json as simplejson
import numpy as np
import optparse
import os
from scipy.sparse import csc_matrix, hstack, lil_matrix
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
import time

def get_keyphrase_data(home, group, max_keyphrases, update=False):
  start = time.time()
  path = '%s/output/entity/%s/keyphrase-data.txt' % (home, group)
  fp = codecs.open(path, 'r', 'UTF-8')
  data = simplejson.load(fp)
  fp.close()
  nrow = data["next-mention-index"]
  ncol = data["next-kp-index"]  
  path = '%s/output/entity/%s/kp-fof-sparse.npz' % (home, group)
  if update or not os.path.exists(path):
    print "Updating first-order features..."
    keyphrase_map = {}
    S = lil_matrix((nrow, ncol))
    for mention_id, mention_index in data['mention-index-map'].iteritems():
      keyphrases = sorted(data['mention-kps'][mention_id], key=lambda x: -x[1])
      keyphrases = keyphrases[:max_keyphrases]
      keyphrase_map[mention_index] = keyphrases
      for kp_index, _, frequency in keyphrases: 
          S[mention_index, kp_index] = 1.0
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(S)
    S = tfidf.transform(S)
    S = S.tocsc()
    sums = S.sum(axis=0)
    np.savez(path, S.data, S.indices, S.indptr, sums)
    fp = codecs.open('%s/output/entity/%s/keyphrase-map.json' % (home, group), 'w', 'UTF-8')
    simplejson.dump(keyphrase_map, fp, indent=4)
    fp.close()
  else: print "Loading keyphrase data..."  
  npzfile = np.load(path)
  S = csc_matrix((npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2']), shape=(nrow, ncol))
  sums = npzfile['arr_3']
  fp = codecs.open('%s/output/entity/%s/keyphrase-map.json' % (home, group), 'r', 'UTF-8')
  keyphrase_map = simplejson.load(fp)
  fp.close()
  keyphrase_map = {int(key):value for key, value in keyphrase_map.iteritems()}
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return data, keyphrase_map, S, sums

def mention_index(sf_data, mention):
    return sf_data['mention-index-map'][mention]

def kp_index(kp_data, kp):
    return kp_data['kp-index-map'][kp]

def get_mentions(kp_data, kp_map, keyphrase_map, i):
    kp = kp_map[i]
    return [j for j in kp_data['kp-mentions'][kp] if i in [index for index, _, _ in keyphrase_map[j]]]

def get_kps(keyphrase_map, i):
    return [kp for kp, _, _ in keyphrase_map[i]]
  
def dump_info(kp_data, home, group):
  nrow = kp_data["next-mention-index"]
  info = {}
  index_map = {}
  labels = range(nrow)
  for key, value in kp_data['mention-index-map'].iteritems():
      folder, doc, occurrence = key.split('::') # js-8::970429-552_UEM::1
      identifier = doc.split('-')[-1][:-4]
      label = folder.split('-')[-1]
      labels[value] = label
      tag = "%s-%s-%s" % (label,identifier, occurrence)
      info[value] = (folder, label, identifier, tag, key, value)
      index_map[tag] = value
  f = open("%s/data/kp-info-%s.csv" % (home, group), 'w')
  for folder, label, identifier, tag, key, value in info.values(): 
      f.write("%s,%s,%s,%s,%s,%d\n" % (folder, label, identifier, tag, key, value))
  f.close()  
  return labels
  
def llr(kp_data, kp_map, keyphrase_map, sums, i1, i2):
    #i1 = sf_index(sf_data, sf1)
    #i2 = sf_index(sf_data, sf2)
    # compute actuaL cell frequencies
    # - outer cells
    ndd = float(kp_data['next-mention-index'])
    npd = float(sums[0,i1])
    ndp = float(sums[0,i2])
    nnd = ndd - npd 
    ndn = ndd - ndp
    # - inner cells
    mentions = [i for i in get_mentions(kp_data, kp_map, keyphrase_map, i1) if i in get_mentions(kp_data, kp_map, keyphrase_map, i2)]
    npp = float(len(mentions))
    npn = npd - npp
    nnp = ndp - npp
    nnn = nnd - nnp
    # compute (randomly) predicted cell frequencies
    enn = nnd * ndn / ndd
    enp = nnd * ndp / ndd
    epn = npd * ndn / ndd
    epp = npd * ndp / ndd
    #print npd, ndp, npp, ndd
    # compute log-likelihood ratio
    result = 0.0
    if nnn > 0: result += nnn * np.log(nnn / enn)
    if nnp > 0: result += nnp * np.log(nnp / enp)
    if npn > 0: result += npn * np.log(npn / epn)
    if npp > 0: result += npp * np.log(npp / epp)
    return 2.0 * result
  
def build_keyphrase_matrix(kp_data, kp_map, keyphrase_map, sums, chi_square_cut_off):
  print "Building keyphrase matrix..."
  nrow2 = ncol2 = kp_data["next-kp-index"]
  S2 = lil_matrix((nrow2, ncol2))
  start = time.time()
  for i in range(nrow2):
      for j in get_mentions(kp_data, kp_map, keyphrase_map, i):
          for k in get_kps(keyphrase_map, j):
              if S2[i,k] > 0: continue
              else:
                  ratio = llr(kp_data, kp_map, keyphrase_map, sums, i, k)
                  if ratio > chi_square_cut_off: 
                      S2[i,k] = ratio
                      S2[k,i] = ratio
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return S2

  
def build_2nd_order_feature_matrix(kp_data, keyphrase_map, S2):
  print "Building 2nd-order feature matrix..."
  start = time.time()
  nrow = kp_data["next-mention-index"]
  ncol = kp_data["next-kp-index"]  
  X = lil_matrix((nrow, ncol))
  for i in range(nrow):
      indices = get_kps(keyphrase_map, i)
      X[i,:] = S2[indices,:].sum(axis=0) / len(indices)
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)  
  return X

def evaluate_kmeans(home, group, labels, data, nclusters, ncomponents, max_keyphrases):
    num_rows, num_cols = data.shape
    if num_cols > ncomponents:
        lsa = TruncatedSVD(ncomponents)
        Y = lsa.fit_transform(data)
        Y = Normalizer(copy=False).fit_transform(Y)
    else: 
        Y = data.todense()
        ncomponents = num_cols
    km = KMeans(n_clusters=nclusters, init='k-means++', max_iter=100, n_init=1,verbose=False)
    km.fit(Y)
    np.savetxt("%s/data/kp-data-%s-%d-%d.csv" % (home, group, ncomponents, max_keyphrases), Y, delimiter=",")
    print("\tHomogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)) 
    print("\tCompleteness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("\tV-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    return km  

def get_features(mention_index, kp_map, keyphrase_map):
  results = []
  for kp_index, _, _ in keyphrase_map[mention_index]:
    kp = kp_map[kp_index]
    results.append(kp)
  return "; ".join(results)
  
def dump_feature_data(home, group, ncomponents, kp_map, keyphrase_map, max_keyphrases, km, labels):
  fp = codecs.open("%s/data/features=by-label-%s-%d-%d.csv" % (home, group, ncomponents, max_keyphrases), 'w', "UTF-8")
  writer = csv.writer(fp, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
  writer.writerow(['Entity', 'Cluster', 'Features', 'Index', 'Mention'])
  keys = mention_map.keys()
  #keys = sorted(keys, key=lambda x: km.labels_[x])
  keys = sorted(keys, key=lambda x: (int(labels[x]), int(km.labels_[x])))
  for key in keys:
    value = mention_map[key]
    #writer.writerow("%d\t%d\t%s\t%d\t%s" % (int(km.labels_[key]), int(labels[key]), get_features(key, coreference_map), int(key), value))
    writer.writerow([int(labels[key]), int(km.labels_[key]), get_features(key, kp_map, keyphrase_map), int(key), value])
  fp.close()     
  
  
if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()

  parser.add_option("-f", help="Use first-order features only", dest='use_first_order', default=False, action='store_true')
  parser.add_option("-s", help="Use second-order features only", dest='use_second_order', default=False, action='store_true')
  parser.add_option("-b", help="Use first- and second-order features", dest='use_both', default=False, action='store_true')
  parser.add_option("-u", help="Update first-order features", dest='update_fof', default=False, action='store_true')
  (opts, args) = parser.parse_args()
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/kb-population.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  group = config.get('clustering','group') 
  max_keyphrases = config.getint('clustering','max-keyphrases') 
  kp_data, keyphrase_map, S, sums = get_keyphrase_data(home, group, max_keyphrases, update=opts.update_fof)
  kp_map = {value:key for key, value in kp_data['kp-index-map'].iteritems()}
  mention_map = {value:key for key, value in kp_data['mention-index-map'].iteritems()}
  labels = dump_info(kp_data, home, group) 
  # 
  chi_square_cut_off = config.getfloat('clustering','chi-square-cut-off')  
  if opts.use_second_order or opts.use_both:
    S2 = build_keyphrase_matrix(kp_data, kp_map, keyphrase_map, sums, chi_square_cut_off)
    X = build_2nd_order_feature_matrix(kp_data, keyphrase_map, S2)
  #
  nclusters = config.getint('clustering','nclusters')  
  ncomponents = config.getint('clustering','ncomponents')  
  if opts.use_first_order:
    print "Evaluating clustering for 1st-order features: max-coreferences = %d; nclusters = %d; ncomponents = %d." % (max_keyphrases, nclusters, ncomponents)
    km = evaluate_kmeans(home, group, labels, S, nclusters, ncomponents, max_keyphrases)
  elif opts.use_both:
    print "Evaluating clustering for both 1st- and 2nd-order features: max-coreferences = %d; nclusters = %d; ncomponents = %d." % (max_keyphrases, nclusters, ncomponents)
    km = evaluate_kmeans(home, group, labels, hstack([S,X]), nclusters, ncomponents, max_keyphrases)
  else:
    print "Evaluating clustering for 2nd-order features: max-coreferences = %d; nclusters = %d; ncomponents = %d." % (max_keyphrases, nclusters, ncomponents)
    km = evaluate_kmeans(home, group, labels, X, nclusters, ncomponents, max_keyphrases)
  dump_feature_data(home, group, ncomponents, kp_map, keyphrase_map, max_keyphrases, km, labels)