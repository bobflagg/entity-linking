import codecs
import ConfigParser
from data.collector import get_output_directory
import numpy as np
import optparse
from clustering.model import HierarchicalClusterModel, KMeansClusterModel
from data.collector import get_output_directory
from domain import SUBTYPES, Entity, EntityCountFeature, Info, Mention
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
 
def summarize_feature_data(cluster, directory, mentions, fp):
  summary = {}
  subtype_weights = {}
  for mention in mentions:
    phrase = mention.info.phrase
    if not phrase in summary: 
      summary[phrase] = {}
      for subtype in SUBTYPES: summary[phrase][subtype] = {}
      subtype_weights[phrase] = {}
      for subtype in SUBTYPES: subtype_weights[phrase][subtype] = 0
    for feature in mention.entity_count_features:
      entity = feature.entity
      entity_map = summary[phrase][entity.type]
      if not entity.phrase in entity_map: entity_map[entity.phrase] = 0
      entity_map[entity.phrase] += 1
      subtype_weights[phrase][entity.type] += 1
  for phrase in summary.keys():
    results = []
    for subtype in sorted(SUBTYPES, key=lambda x: -subtype_weights[phrase][x]):
      entity_map = summary[phrase][subtype]
      keys = sorted(entity_map.keys(), key=lambda x: -entity_map[x])
      result = "; ".join("%s:%d" % (key, entity_map[key]) for key in keys)
      results.append("%s: %s" % (subtype, result))
    fp.write("%d\t%s\t%s\n" % (cluster, phrase, "\t".join(results)))
 
def summarize_keyphrase_feature_data(cluster, directory, mentions, fp):
  summary = {}
  subtype_weights = {}
  for mention in mentions:
    phrase = mention.info.phrase
    if not phrase in summary: 
      summary[phrase] = {}
      for subtype in SUBTYPES: summary[phrase][subtype] = {}
      subtype_weights[phrase] = {}
      for subtype in SUBTYPES: subtype_weights[phrase][subtype] = 0
    for feature in mention.entity_count_features:
      entity = feature.entity
      entity_map = summary[phrase][entity.type]
      if not entity.phrase in entity_map: entity_map[entity.phrase] = 0
      entity_map[entity.phrase] += 1
      subtype_weights[phrase][entity.type] += 1
  for phrase in summary.keys():
    results = []
    for subtype in sorted(SUBTYPES, key=lambda x: -subtype_weights[phrase][x]):
      entity_map = summary[phrase][subtype]
      keys = sorted(entity_map.keys(), key=lambda x: -entity_map[x])
      result = "; ".join("%s:%d" % (key, entity_map[key]) for key in keys)
      results.append("%s: %s" % (subtype, result))
    fp.write("%d\t%s\t%s\n" % (cluster, phrase, "\t".join(results)))

def dump_feature_data(mentions, labels, output_directory):
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
   summarize_feature_data(key, output_directory, cluster_map[key], fp)
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
    cmodel = ClusterModel(n_clusters)
  cmodel.fit(data)
  # - build and store feature report
  entities = get_entities(output_directory)
  info_data = filter_info_data(indices, output_directory)
  mentions = filter_mention_data(indices, entities, info_data, output_directory)
  dump_feature_data(mentions, cmodel.labels(), output_directory)
  finish = time.time()
  print 'total processing time: %0.3f s' % (finish-start)

# python collector.py /home/disambiguation/config/js.cfg -m -r -s
# python collector.py /home/disambiguation/config/jr.cfg -m -r -s
# python disambiguator.py /home/disambiguation/config/jr.cfg
# python disambiguator.py /home/disambiguation/config/js.cfg

