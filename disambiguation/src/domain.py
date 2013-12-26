import codecs

class Enum(set):
    def __getattr__(self, name):
        if name in self: return name
        raise AttributeError  
  
SUBTYPES = Enum("COMPANY CRIMINAL EDUCATIONAL GEO GOVERNMENT HOSTNAME LOCATION ORGANIZATION PERFORMING PERSON SPORTS".split())

class Entity(object):
  def __init__(self, index, line):
    etype, phrase = line.strip().split('\t')
    self.index = index
    self.type = etype.strip()
    self.phrase = phrase.strip()
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "[%d] %s (%s)" % (self.index, self.phrase, self.type)

class Info(object):
  def __init__(self, index, document_name, document_index, phrase, position, subtype=SUBTYPES.PERSON):
    self.index = index
    self.document_name = document_name
    self.document_index = document_index
    self.phrase = phrase
    self.position = position
    self.subtype = subtype
   
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "[%d] %s\t%s [%d]" % (self.index, self.phrase, self.document_name, self.document_index)

class EntityCountFeature(object):
  def __init__(self, entity, count):
    self.entity = entity
    self.count = count
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "%s -->> %d" % (self.entity, int(self.count))
    
class EntityProximityFeature(object):
  def __init__(self, entity, proximity):
    self.entity = entity
    self.proximity = proximity
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "%s -->> %.2f" % (self.entity, self.proximity)
    
class Mention(object):

  def __init__(self, entities, info, fdata):
    #n_entities = len(entities)
    self.info = info
    self.entity_count_features = []
    self.entity_proximity_features = []
    for data in fdata:
      t, j, value = data.split(":")
      j = int(j)
      value = float(value)
      if t == 'p': 
        self.entity_proximity_features.append(EntityProximityFeature(entities[j], value))
      else:
        self.entity_count_features.append(EntityCountFeature(entities[j], value))

  def set_keyphrase_info(self, indices_map):
    self.keyphrase_indices = indices_map[self.info.document_name]

  def set_topic_info(self, corpus_transformed):
    self.topic_distribution = corpus_transformed[self.info.document_index]

  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    results = ["%s" % feature for feature in self.entity_count_features]
    results.extend("%s" % feature for feature in self.entity_proximity_features)
    return "%s:\n\t%s" % (self.info, "\n\t".join(results))
        
  def get_phrase(self):
        return self.info[index]

class Topic(object):
  def __init__(self, index, model, n_top=3):
    self.index = index
    self.words = []
    self.probs = []
    for prob, word in model.show_topic(index)[:n_top]:
      self.words.append(word)
      self.probs.append(prob)
   
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return "{%s}" % ", ".join(self.words)

