'''
Classes to support ETL.
'''

class Enum(set):
    def __getattr__(self, name):
        if name in self: return name
        raise AttributeError  
  
SUBTYPES = Enum("COMPANY CRIMINAL EDUCATIONAL GEO GOVERNMENT HOSTNAME LOCATION ORGANIZATION PERFORMING PERSON SPORTS".split())

class Mention(object):
  MENTIONS = []
  NEXT_INDEX = 0

  def __init__(self, doc_id, doc_name, encoding, context, subtype=SUBTYPES.PERSON):
    self.doc_id = doc_id
    self.doc_name = doc_name
    self.context = context
    phrase, position = encoding.split(":")
    self.subtype = subtype
    self.phrase = phrase
    self.position = int(position)
    self.featureSets = []
  
  def add_featureSets(self, feature_data):  
    for item in feature_data: 
      subtype = item.split(':')[0]
      subtype = subtype.strip().upper()
      if subtype in SUBTYPES: self.featureSets.append(FeatureSet(item))
    
  def info(self):
    results = ["%s\t%s\t%s\t%d" % (self.doc_name, self.subtype, self.phrase, self.position)]
    return "\t".join(results)
    
  def record(self):
    results = []
    for featureSet in self.featureSets: results.append(featureSet.record())
    return "\t".join(results)
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    results = ["%s:%d" % (self.phrase, self.position)]
    results.extend("%s" % featureSet for featureSet in self.featureSets)
    return "\t".join(results)
  
class FeatureSet(object):
  
  def __init__(self, item_data): # u'PERSON:Brent Cann:0.777777777778:1'
    subtype, phrase, proximity, freq = item_data.split(':')
    self.entity = Entity.get_entity(subtype, phrase)
    self.subtype = self.entity.subtype
    self.proximity = float(proximity)
    self.freq = int(freq)

  def __lt__(self, other):
    return self.entity <  other.entity
    
  def record(self):
    index = self.entity.index
    return "p:%d:%.2f\tc:%d:%d" % (index, self.proximity, index, self.freq)
    
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return u"%s:%.4f:%d" % (self.entity, self.proximity, self.freq)

class Entity(object):
  ENTITIES = []
  NEXT_INDEX = 0
  MAP = {}
    
  @classmethod
  def get_entity(cls, subtype, phrase):
    phrase = phrase.strip().lower()
    subtype = subtype.strip().upper()
    key = "%s:%s" % (subtype, phrase)
    if key in Entity.MAP: return cls.MAP[key]
    entity = Entity(subtype, phrase, cls.NEXT_INDEX)
    cls.ENTITIES.append(entity)
    cls.MAP[key] = entity
    cls.NEXT_INDEX += 1
    return entity
  
  def __init__(self, subtype, phrase, index):
    self.phrase = phrase.strip().lower()
    self.subtype = subtype.strip().upper()
    self.key = "%s:%s" % (self.subtype, self.phrase)
    self.index = index

  def record(self):
    return "%s\t%s" % (self.subtype, self.phrase)

  def __lt__(self, other):
    if self.subtype ==  other.subtype: return self.phrase < other.phrase
    return self.subtype <  other.subtype
  
  def __str__(self):
    return unicode(self).encode('utf-8')
  
  def __unicode__(self):
    return u"%s:%s" % (self.subtype, self.phrase)