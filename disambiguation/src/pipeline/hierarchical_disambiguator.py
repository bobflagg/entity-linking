from data.corpus_builder import build_context_corpus

def get_context_features(home, directory, target):
  path = "%s/%s/%s-context" % (home, directory, target)
  data = build_context_corpus(path)