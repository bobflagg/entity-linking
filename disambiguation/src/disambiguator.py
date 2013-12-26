import codecs
import ConfigParser
from data.collector import get_output_directory
from gensim import corpora, models
import numpy as np
import optparse
from clustering.model import HierarchicalClusterModel, KMeansClusterModel
from data.collector import get_output_directory
from domain import SUBTYPES, Entity, EntityCountFeature, Info, Mention, Topic
import time

def get_data(output_directory):
  npzfile = np.load("%s/data.npz" % output_directory)
  data = npzfile['arr_0']
  indices = npzfile['arr_1']
  return indices, data

def get_entities(output_directory):
  print "\tgetting entities..."
  start = time.time()
  entities = []
  index = 0
  for line in codecs.open("%s/entity" % output_directory, 'r', 'utf-8'):
    entities.append(Entity(index, line))
    index += 1
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return entities

def get_keyphrase_data(output_directory):
  print "\tgetting keyphrase data..."
  start = time.time()
  indices_map = {}
  for line in codecs.open("%s/keyphrase-indices" % output_directory, 'r', 'utf-8'):
    parts = line.strip().split('\t')
    doc = parts[0]
    indices = [int(item) for item in parts[1][1:-1].split(', ')]
    indices_map[doc] = indices
  index_map = {}
  for line in codecs.open("%s/keyphrase-index" % output_directory, 'r', 'utf-8'):
    parts = line.strip().split('\t')
    index_map[int(parts[1].strip())] = parts[0].strip()
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return index_map, indices_map

def get_info(output_directory):
  fp = codecs.open("%s/info" % output_directory, 'r', 'utf-8')
  current_doc = None
  doc_index = 0
  for record in fp:
    parts = record.strip().split("\t")
    doc = parts[1]
    if current_doc and not current_doc == doc: doc_index += 1
    current_doc = doc
    yield Info(int(parts[0]), parts[1], doc_index, parts[3], int(parts[4]), parts[2])
  fp.close()

def filter_info_data(indices, output_directory):
  print "\tfiltering info data..."
  start = time.time()
  data = [info for info in get_info(output_directory) if info.index in indices]
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return data

def filter_keyphrase_data(indices, output_directory):
  print "\tfiltering keyphrase data..."
  start = time.time()
  data = [info for info in get_info(output_directory) if info.index in indices]
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return data
        
def filter_mention_data(indices, entities, info_data, output_directory):
  print "\tfiltering mention data..."
  start = time.time()
  path = "%s/mention" % output_directory
  mentions = []
  position = 0
  for line in codecs.open(path, 'r', 'UTF-8'):
    parts = line.strip().split('\t')
    index = int(parts[0])
    if index in indices:
      mentions.append(Mention(entities, info_data[position], parts[1:]))
      position += 1  
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return mentions
        
def load_topic_model_components(output_directory):
  print "\tloading topic model components..."
  start = time.time()
  dictionary = corpora.Dictionary.load('/%s/document.dict' % output_directory)
  lda = models.ldamodel.LdaModel.load('/%s/document.lda' % output_directory)
  corpus_lda = corpora.MmCorpus('/%s/document-lda.mm' % output_directory)  
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
  return dictionary, lda, corpus_lda

def append_topic_info(mentions, corpus_transformed):
  print "\tappending topic info..."
  start = time.time()
  for mention in mentions: mention.set_topic_info(corpus_transformed)
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)

def append_keyphrase_info(mentions, indices_map):
  print "\tappending keyphrase info..."
  start = time.time()
  for mention in mentions: mention.set_keyphrase_info(indices_map)
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)
 
def summarize_feature_data(cluster, directory, mentions, keyphrase_index_map, topic_model, fp):
  summary = {}
  subtype_weights = {}
  for mention in mentions:
    phrase = mention.info.phrase
    if not phrase in summary: 
      summary[phrase] = {}
      for subtype in SUBTYPES: summary[phrase][subtype] = {}
      summary[phrase]['keyphrase'] = {}
      summary[phrase]['topic'] = {}
      subtype_weights[phrase] = {}
      for subtype in SUBTYPES: subtype_weights[phrase][subtype] = 0
    # entity co-occurrence
    for feature in mention.entity_count_features:
      entity = feature.entity
      entity_map = summary[phrase][entity.type]
      if not entity.phrase in entity_map: entity_map[entity.phrase] = 0
      entity_map[entity.phrase] += 1
      subtype_weights[phrase][entity.type] += 1
    # keyphrase
    for index in mention.keyphrase_indices:
      if not (index in summary[phrase]['keyphrase']): summary[phrase]['keyphrase'][index] = 0
      summary[phrase]['keyphrase'][index] += 1
    # topic
    for topic, weight in mention.topic_distribution:
      if not (topic in summary[phrase]['topic']): summary[phrase]['topic'][topic] = 0
      summary[phrase]['topic'][topic] += weight
  for phrase in summary.keys():
    results = []
    # entity co-occurrence
    for subtype in sorted(SUBTYPES, key=lambda x: -subtype_weights[phrase][x]):
      entity_map = summary[phrase][subtype]
      keys = sorted(entity_map.keys(), key=lambda x: -entity_map[x])
      result = "; ".join("%s:%d" % (key, entity_map[key]) for key in keys)
      results.append("%s: %s" % (subtype, result))
    # keyphrase
    keys = sorted(summary[phrase]['keyphrase'].keys(), key=lambda x: -summary[phrase]['keyphrase'][x])
    result = "; ".join("%s:%d" % (keyphrase_index_map[key], summary[phrase]['keyphrase'][key]) for key in keys)
    results.append(result)
    # topic
    keys = sorted(summary[phrase]['topic'].keys(), key=lambda x: -summary[phrase]['topic'][x])
    result = "; ".join("%d:%s" % (key, Topic(key, topic_model)) for key in keys)
    results.append(result)
    fp.write("%d\t%s\t%s\n" % (cluster, phrase, "\t".join(results)))

def dump_feature_data(mentions, labels, output_directory, keyphrase_index_map, topic_model):
  print "\tdumping feature data..."
  start = time.time()
  cluster_map = {}
  position = 0
  for mention in mentions: 
    cluster = labels[position]
    if not cluster in cluster_map: cluster_map[cluster] = set()
    cluster_map[cluster].add(mention)
    position += 1
  fp = codecs.open("%s/cluster-feature-summary.tsv" % output_directory, 'w', 'utf-8')
  for key in cluster_map.keys():
   summarize_feature_data(key, output_directory, cluster_map[key], keyphrase_index_map, topic_model, fp)
  fp.close()  
  finish = time.time()
  print '\ttook %0.3f s' % (finish-start)

if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()
  (opts, args) = parser.parse_args()
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/js.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  data_directory = config.get('main', 'data-directory')
  target = config.get('main','target')
  surface_form = config.get('main','surface-form')
  clustering_type = config.get('clustering','clustering-type')
  n_clusters = config.getint('clustering','n-clusters')
  print("Clustering %s data using %s clustering" % (target, clustering_type))
  start = time.time()
  # - get the relevant mention data
  output_directory = get_output_directory(home, target)
  indices, data = get_data(output_directory)
  # - fit a clustering model to the data
  if clustering_type == 'hierarchical': 
    threshold = config.getfloat('clustering','threshold')
    cmodel = HierarchicalClusterModel(threshold)
  else:
    n_clusters = config.getint('clustering','n-clusters')
    cmodel = KMeansClusterModel(n_clusters)
  cmodel.fit(data)
  # build and store feature report
  # - start with entity collocation features
  entities = get_entities(output_directory)
  info_data = filter_info_data(indices, output_directory)
  mentions = filter_mention_data(indices, entities, info_data, output_directory)
  # - add keyphrase features
  index_map, indices_map = get_keyphrase_data(output_directory)
  append_keyphrase_info(mentions, indices_map)
  # - add topic model features
  dictionary, lda, corpus_lda = load_topic_model_components(output_directory)
  append_topic_info(mentions, corpus_lda)
  # - dump features
  dump_feature_data(mentions, cmodel.labels(), output_directory, index_map, lda)
  finish = time.time()
  print 'total processing time: %0.3f s' % (finish-start)

# python collector.py /home/disambiguation/config/js.cfg -m -r -s
# python collector.py /home/disambiguation/config/jr.cfg -m -r -s
# python disambiguator.py /home/disambiguation/config/jr.cfg
# python disambiguator.py /home/disambiguation/config/js.cfg

