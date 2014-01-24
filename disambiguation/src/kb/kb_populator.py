'''
1. update your Python path:
  export PYTHONPATH=/home/disambiguation/entity-linking/disambiguation/src
  export PYTHONPATH=/opt/disambiguation/entity-linking/disambiguation/src
2. move to the directory containing the collector script:
  cd /home/disambiguation/entity-linking/disambiguation/src/kb
  cd /opt/disambiguation/entity-linking/disambiguation/src/kb
3. run the script:
  python kb_populator.py <path to config file> [options]
for example:
  python kb_populator.py /home/disambiguation/config/kb-population.cfg -t
Available options are:
  
    -k          Update keyphrase features
    -c          Update coreference data
    -t          Update document topic model
    -v          Update context vector space model

The configuration file is pretty simple and the only thing that you'll probably want
to change is the search-pattern entry in the main section, which specifies the directories
below
	/home/disambiguation/corpus/entity
to process.  Currently it is set with
	search-pattern: js-*
so all the John Smith directories are processed.

'''
import codecs
import ConfigParser
from data.corpus_builder import DirectoryBackedCorpus, LabeledCorpus
from data.domain import SUBTYPES
from domain import Topic
from gensim import corpora, models
import glob
import json as simplejson
from kpe.key_concept_extractor import extract_kc
from nltk.tokenize import sent_tokenize, word_tokenize
import optparse
import os
import time

def build_document_topic_model(home, config, update=False):
  '''
  Builds and stores an LDA topic model for the ambient document corpus.
  '''
  path = '%s/output/ambient-document.lda' % home
  if update or not os.path.exists(path):
    print "Building and storing ambient document topic model."
    start = time.time()
    corpus_directory = config.get('document-topic-model','directory')
    corpus_path = "%s/corpus/%s" % (home, corpus_directory)
    description = "Ambient document corpus"
    no_below = config.getint('document-topic-model','no-below')
    no_above = config.getfloat('document-topic-model','no-above')
    corpus = DirectoryBackedCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('%s/output/ambient-document.dict' % home)
    corpora.MmCorpus.serialize('%s/output/ambient-document.mm' % home, corpus)
    num_topics = config.getint('document-topic-model','num-topics')
    lda = models.ldamodel.LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
    lda.save(path)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "Ambient document topic model appears to be up-to-date."

def build_group_topic_model(home, config, g, update=False):
  '''
  Builds and stores an LDA topic model for the group document corpus.
  '''
  path = '%s/output/entity/%s/document.lda' % (home, g)
  if update or not os.path.exists(path):
    print "Building and storing group topic model."
    start = time.time()
    corpus_path = "%s/corpus/entity/%s" % (home, g)
    description = "group topic model"
    no_below = config.getint('document-topic-model','no-below')
    no_above = config.getfloat('document-topic-model','no-above')
    corpus = LabeledCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('%s/output/entity/%s/document.dict' % (home, g))
    corpora.MmCorpus.serialize('%s/output/entity/%s/document.mm' % (home, g), corpus)
    num_topics = config.getint('document-topic-model','num-topics')
    lda = models.ldamodel.LdaModel(corpus, id2word=corpus.dictionary, num_topics=num_topics, passes=12)
    lda.save(path)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "Group topic model appears to be up-to-date."

def build_context_vector_space_model(home, config, update=False):
  '''
  Builds and stores a TF-IDF vector space model for the ambient context corpus.
  '''
  path = '%s/output/ambient-context-vector-space-model.mm' % home
  if update or not os.path.exists(path):
    print "Building and storing ambient context vector space model."
    start = time.time()
    corpus_directory = config.get('context-vector-space-model','directory')
    corpus_path = "%s/corpus/%s" % (home, corpus_directory)
    description = "Ambient context corpus"
    no_below = config.getint('context-vector-space-model','no-below')
    no_above = config.getfloat('context-vector-space-model','no-above')
    corpus = DirectoryBackedCorpus(corpus_path, description, no_below=no_below, no_above=no_above)
    corpus.initialize_dictionary()
    corpus.dictionary.save('%s/output/ambient-context.dict' % home)
    corpora.MmCorpus.serialize('%s/output/ambient-context.mm' % home, corpus)
    tfidf = models.TfidfModel(corpus)
    tfidf.save(path)
    finish = time.time()
    print '\ttook %0.3f s' % (finish-start)
  else:
    print "Ambient context vector space model appears to be up-to-date."

def collect_entity_coference_data(directory, sf_data):
  '''
  Collects and stores all surface forms appearing in feature entity kb files.
  '''
  path = "%s/features-entities.json" % directory
  fp = codecs.open(path, 'r', 'UTF-8')
  data = simplejson.load(fp)
  doc_ids = set()
  for item in data:
    subtype = item['subtype']
    doc_id = item['doc_id']
    doc_ids.add(doc_id) 
    if not doc_id in sf_data['doc-sfs']: 
      sf_data['doc-sfs'][doc_id] = set()
      sf_data['doc-index-map'][doc_id] = sf_data['next-doc-index']
      sf_data['next-doc-index'] += 1
    if subtype in SUBTYPES: 
      cofereference = "%s::%s" % (subtype, item['entity'])
      if not cofereference in sf_data['sf-index-map']: 
        sf_data['sf-docs'][cofereference] = set()
        sf_data['sf-index-map'][cofereference] = sf_data['next-sf-index']
        sf_data['next-sf-index'] += 1
      sf_data['doc-sfs'][doc_id].add(sf_data['sf-index-map'][cofereference])
      sf_data['sf-docs'][cofereference].add(sf_data['doc-index-map'][doc_id])
  fp.close()
  ndocs = len(list(doc_ids))
  return ndocs

def collect_coference_data(home, config, update=False):
  '''
  Collects and stores surface form for all specified (with a search pattern) entities.
  '''
  entity_groups = config.get('main','entity-groups').split(',')
  for g in entity_groups:
    d = "%s/output/entity/%s" % (home, g)
    path = '%s/coreference-data.txt' % d 
    if update or not os.path.exists(path):
      print "Collecting and storing KB coreference data for %s." % g
      start = time.time()
      sf_data = {'next-sf-index':0, 'sf-index-map':{}, 'next-doc-index':0, 'doc-index-map':{}, 'doc-sfs':{}, 'doc-topics':{}, 'sf-docs':{}}
      ndocs = 0
      for subd in os.listdir(d):
        full_path = "%s/%s" % (d, subd)
        if os.path.isdir(full_path): ndocs += collect_entity_coference_data(full_path, sf_data)
      print "ndocs: %d" % ndocs
      for doc_id in sf_data['doc-sfs'].keys():
        sfs = list(sf_data['doc-sfs'][doc_id])
        sfs.sort()
        sf_data['doc-sfs'][doc_id] = sfs
      for cofereference in sf_data['sf-docs'].keys():
        docs = list(sf_data['sf-docs'][cofereference])
        docs.sort()
        sf_data['sf-docs'][cofereference] = docs
      fp = codecs.open(path, 'w', 'UTF-8')
      simplejson.dump(sf_data, fp, indent=4)
      fp.close()
      finish = time.time()
      print '\ttook %0.3f s' % (finish-start)
    else:
      print "Coreference data appears to be up-to-date."

def extract_topic_features(output_directory, documents , dictionary, model, tolerance):
  '''
  Extracts and stores topic features from the given documents, using the
  provided dictionary and topic model.
  '''
  print "\t- topic features."
  corpus = SimpleCorpus(documents , dictionary)
  ndocs = len(documents)
  #feature_data = {'number-of-documents':ndocs, 'topics':{}}
  map = {}
  for document in model[corpus]:
    for topic, weight in document: 
      if weight > tolerance:
        if not topic in map: map[topic] = 0.0
        #map[topic].append(weight)
        map[topic] += weight
  #for key in map.keys(): map[key] = sorted(map[key], key=lambda x: -x)
  data = []
  for key in map.keys(): data.append({"topicid":key, "score":map[key] / ndocs})
  #data = {'number-of-documents':ndocs, 'topics':map}
  fp = codecs.open("%s/features-topics.json" % output_directory, 'w', 'UTF-8')
  simplejson.dump(data, fp, indent=4)
  fp.close()

from data.domain import FeatureSet
import re
def extract_entity_features(output_directory, surface_form_re, f='Proximity_AllCollapsedContext_JS'):
  '''
  Extracts and stores entity features from the given file.
  '''
  print "\t- entity features."
  label = output_directory.split('/')[-1]
  mentions_path = '%s/%s' % (output_directory, f)
  mentions_data = codecs.open(mentions_path, 'r', 'UTF-8').readlines()
  features = []
  finished_doc = True
  for line in mentions_data: 
    line = line.strip()
    if line:
      if finished_doc: 
        doc_id, path = line.split(":")
        doc_id = doc_id.strip()
        doc_id = "%s::%s" % (label,doc_id)
        #path = path.strip()[1:-1]
        #doc_name = path.split('/')[-1]
        finished_doc = False
      else:
        data = line.split('\t')
        sf = data[0].split(':')[0]
        if surface_form_re.search(sf): 
          for item in data[1:-1]:
            fs = FeatureSet(doc_id, item)
            if not surface_form_re.search(fs.entity.phrase): features.append(fs.to_dict())
    else: 
      finished_doc = True
  fp = codecs.open("%s/features-entities.json" % output_directory, 'w', 'UTF-8')
  simplejson.dump(features, fp, indent=4)
  fp.close()

def extract_keyphrase_features(output_directory, documents, max_keyphrases, update=False):
  '''
  Extracts and stores keyphrase features from the given documents.
  '''
  path = "%s/features-keyphrases.json" % output_directory
  if update or not os.path.exists(path):
    print "\t- keyphrase features."
    ndocs = len(documents)
    map = {}
    for document in documents:
      keyphrases = extract_kc(document)
      for keyphrase in keyphrases:
        if not keyphrase in map: map[keyphrase] = 0
        map[keyphrase] += 1
    #data = {'number-of-documents':ndocs, 'keyphrases':map}
    data = []
    for key, value in map.iteritems(): data.append({"keyphrase":key, "score":value})
    fp = codecs.open(path, 'w', 'UTF-8')
    simplejson.dump(data, fp, indent=4)
    fp.close()
  else:
    print "\t- keyphrase features appear to be up-to-data."

class SimpleCorpus(object):
  def __init__(self, docs, dictionary):
    self.docs = docs
    self.dictionary = dictionary

  def __iter__(self):
    for doc in self.docs:
      sentences = sent_tokenize(doc.lower())
      tokenized_sentences = [word_tokenize(x) for x in sentences] 
      tokens = sum(tokenized_sentences, [])
      yield self.dictionary.doc2bow(tokens)

def extract_entity_kb_data(home, config, opts, doc_dictionary, doc_model, group, directory, default_surface_form_re):
  print " ->> Extracting features from directory ~/%s/%s." % (group, directory)
  #surface_form_re = get_surface_forms("%s/corpus/entity/%s/%s" % (home, group, directory), default_surface_form_re)
  surface_form_re = get_surface_forms("%s/output/entity/%s/%s" % (home, group, directory), default_surface_form_re)
  kb_entry = {}
  output_directory = "%s/output" % home
  target_directory = "%s/entity/%s/%s" % (output_directory, group, directory)
  if not os.path.exists(target_directory): os.makedirs(target_directory)
  # get entity documents
  _, documents = get_documents(target_directory)
  # topic features
  tolerance = config.getfloat('document-topic-model','feature-tolerance')
  extract_topic_features(target_directory, documents, doc_dictionary, doc_model, tolerance)
  # keyphrase features
  max_keyphrases= config.getint('keyphrase','max-keyphrases')
  extract_keyphrase_features(target_directory, documents, max_keyphrases, opts.update_keyphrase_features)
  #extract_keyphrase_features(target_directory, documents, max_keyphrases, True)
  # entity features
  #extract_entity_features(target_directory, f='Proximity_AllCollapsedContext_JS', surface_forms = ['Smith', '^John$'])
  extract_entity_features(target_directory, surface_form_re)

def get_documents(target_directory):
  path = "%s/Normalized_JS" % target_directory
  ids = []
  documents = []
  for d in [f for f in os.listdir(path) if f.endswith("_NORM")]:
    ids.append(d)
    documents.append(codecs.open("%s/%s" % (path, d), 'r', 'UTF-8').read())
  return ids, documents
  
def extract_kb_data(home, config, opts):
  print "Extracting features."
  output_directory = "%s/output" % home
  doc_dictionary = corpora.Dictionary.load('%s/ambient-document.dict' % output_directory)
  doc_model = models.ldamodel.LdaModel.load('%s/ambient-document.lda' % output_directory)
  # process all entities
  entity_groups = config.get('main','entity-groups').split(',')
  for g in entity_groups:
    path = "%s/corpus/entity/%s" % (home, g)
    default_surface_form_re = get_surface_forms(path)
    path = "%s/output/entity/%s" % (home, g)
    for d in os.listdir(path):
      full_path = "%s/%s" % (path, d)
      if os.path.isdir(full_path):
        extract_entity_kb_data(home, config, opts, doc_dictionary, doc_model, g, d, default_surface_form_re)
  
def collect_topic_data(home, config, opts):
  output_directory = "%s/output" % home
  #dictionary = corpora.Dictionary.load('%s/ambient-document.dict' % output_directory)
  #model = models.ldamodel.LdaModel.load('%s/ambient-document.lda' % output_directory)
  entity_groups = config.get('main','entity-groups').split(',')
  for g in entity_groups:
    dictionary = corpora.Dictionary.load('%s/entity/%s/document.dict' % (output_directory,g))
    model = models.ldamodel.LdaModel.load('%s/entity/%s/document.lda' % (output_directory,g))
    path = "%s/output/entity/%s/topic-data.json" % (home, g) 
    if opts.update_topic_data or not os.path.exists(path):
      print "Collecting topic data for group %s." % g
      start = time.time()
      topic_data = {
          'number-of-topics':model.num_topics,
          'topic_description':{},
          'document_topics':{}
      }
      directory = "%s/output/entity/%s" % (home, g)
      for d in os.listdir(directory):
        full_path = "%s/%s" % (directory, d)
        if os.path.isdir(full_path):
          collect_group_topic_data(home, dictionary, model, g, d, topic_data)
      fp = codecs.open(path, 'w', 'UTF-8')
      simplejson.dump(topic_data, fp, indent=4)
      fp.close()
      finish = time.time()
      print '\ttook %0.3f s' % (finish-start)
    else:
      print "Topic data appears to be up-to-date."

def collect_group_topic_data(home, dictionary, model, g, d, topic_data):
  '''
  Collects and stores topic data for the entity corresponding to the given sub-directory and
  group.
  '''
  target_directory = "%s/output/entity/%s/%s" % (home, g, d)
  ids, documents = get_documents(target_directory)
  corpus = SimpleCorpus(documents, dictionary)
  map = {}
  for index, document in enumerate(model[corpus]):
    for t, _ in document:
      if not t in topic_data['topic_description']: 
        topic = Topic(t, model)
        topic_data['topic_description'][t] = "%s" % topic
    map[ids[index]] = [(t, w, topic_data['topic_description'][t]) for t, w in document]
  topic_data['document_topics'][d] = map 

def get_surface_forms(path, default_surface_form_re=None):
  path = "%s/surface-form.txt" % path
  if os.path.exists(path):
    fp = codecs.open(path, 'r', 'UTF-8')
    surface_forms = []
    for line in fp: surface_forms.append(line.strip())
    fp.close()
    surface_form_pattern = "|".join(surface_forms)
    surface_form_re = re.compile(surface_form_pattern, re.IGNORECASE)
  else: surface_form_re = default_surface_form_re
  return surface_form_re

if __name__ == "__main__":
  # Parse command-line options.
  parser = optparse.OptionParser()

  parser.add_option("-b", help="Skip extracting KB data", dest='skip_extracting_kb_data', default=False, action='store_true')
  parser.add_option("-k", help="Update keyphrase features", dest='update_keyphrase_features', default=False, action='store_true')
  parser.add_option("-c", help="Update coreference data", dest='update_coreference_data', default=False, action='store_true')
  parser.add_option("-t", help="Update document topic model", dest='update_topic_model', default=False, action='store_true')
  parser.add_option("-v", help="Update context vector space model", dest='update_vector_model', default=False, action='store_true')
  parser.add_option("-x", help="Update topic data", dest='update_topic_data', default=False, action='store_true')
  (opts, args) = parser.parse_args()
  # Read configuration info:
  if len(args) == 1: config_path = args[0]
  else: config_path = "/home/disambiguation/config/kb-population.cfg"
  config = ConfigParser.ConfigParser()
  config.read(config_path) 
  home = config.get('main','home')
  # Build ambient document corpus topic model
  build_document_topic_model(home, config, update=opts.update_topic_model)
  build_group_topic_model(home, config, "js", update=True)
  # Build ambient context corpus vector space model
  build_context_vector_space_model(home, config, update=opts.update_vector_model)
  # Extract features
  if not opts.skip_extracting_kb_data: extract_kb_data(home, config, opts)
  # Collect coference data
  collect_coference_data(home, config, update=opts.update_coreference_data)  
  # Collect topic data
  collect_topic_data(home, config, opts)