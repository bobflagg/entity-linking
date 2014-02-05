import codecs
import ConfigParser
from data.domain import SUBTYPES
import json as simplejson
import optparse
import os
import pickle
import time

from domain import KeyphraseDataset

def collect_entity_data(directory, dataset):
  '''
  Collects and stores keyphrases appearing in feature entity kb files.
  '''
  path = "%s/features-entities.json" % directory
  fp = codecs.open(path, 'r', 'UTF-8')
  data = simplejson.load(fp)
  for item in data:
    # get the corresponding mention
    doc_id = item['doc_id']
    occurrence = item['occurrence']
    mention = dataset.get_mention(doc_id, occurrence)
    subtype = item['subtype']    
    if subtype in SUBTYPES: 
      dataset.add_keyphrase(mention, item['entity'], item['proximity'], item['frequency'])
  fp.close()

def collect_group_data(home, config, g):
  '''
  Collects entity keyphrase data for the given group.
  '''
  d = "%s/output/entity/%s" % (home, g)
  # path = '%s/kore-keyphrase-data.txt' % d 
  start = time.time()
  dataset = KeyphraseDataset(g)
  for subd in os.listdir(d):
    full_path = "%s/%s" % (d, subd)
    if os.path.isdir(full_path): 
      collect_entity_data(full_path, dataset)
  #fp = codecs.open(path, 'wb', 'UTF-8')
  #pickle.dump(dataset, fp)
  #fp.close()
  finish = time.time()
  print 'Loading group %s data took %0.3f s' % (g, finish-start)
  return dataset    

if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()
  parser.add_option("-u", help="Update data", dest='update_data', default=False, action='store_true')
  (opts, args) = parser.parse_args()
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/kb-population.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  # Collect keyphrase data
  #collect_data(home, config, update=opts.update_data)  
