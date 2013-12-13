from sklearn.feature_extraction.text import TfidfVectorizer

#http://scikit-learn.org/stable/modules/feature_extraction.html
def tf_idf_features(corpus):
  vectorizer = TfidfVectorizer(min_df=2)
  X = vectorizer.fit_transform(corpus.documents)
  return vectorizer
  